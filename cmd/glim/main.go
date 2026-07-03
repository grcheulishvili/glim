package main

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"glim/internal/engine"
	"glim/internal/inference"
	"glim/internal/mask"

	"github.com/spf13/pflag"
)

func parseSchemaFlag(schemaRaw string) map[string]string {
	schema := make(map[string]string)
	if schemaRaw == "" {
		return schema
	}

	pairs := strings.Split(schemaRaw, ",")
	for _, pair := range pairs {
		kv := strings.SplitN(pair, ":", 2)
		if len(kv) == 2 {
			schema[strings.TrimSpace(kv[0])] = strings.TrimSpace(kv[1])
		} else if len(kv) == 1 {
			schema[strings.TrimSpace(kv[0])] = "string"
		}
	}
	return schema
}

func main() {
	var (
		modelPath   string
		ctxLength   int
		numThreads  int
		schemaRaw   string
		temperature float32
		maxTokens   int
		verbose     bool
	)

	homeDir, _ := os.UserHomeDir()
	defaultModelPath := fmt.Sprintf("%s/.local/share/glim/models/qwen2.5-1.5b-instruct-q4_k_m.gguf", homeDir)

	pflag.StringVarP(&modelPath, "model", "m", defaultModelPath, "Path to GGUF target model binary")
	pflag.IntVarP(&ctxLength, "ctx-size", "c", 2048, "KV matrix context token allocation limit")
	pflag.IntVarP(&numThreads, "threads", "t", 4, "Physical CPU execution threads allocated for compute")
	pflag.StringVarP(&schemaRaw, "json", "j", "ip:string,vector:string", "Comma separated target schema mapping (key:type)")
	pflag.IntVarP(&maxTokens, "max-tokens", "n", 128, "Max token window size bound per pipeline evaluation sequence")
	pflag.Float32Var(&temperature, "temp", 0.0, "Sampling temperature modifier")
	pflag.BoolVarP(&verbose, "verbose", "v", false, "Enable verbose internal diagnostic logging from llama.cpp")
	pflag.Parse()

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("[!] Engine Matrix file missing at target path layout.\nTarget: %s\n", modelPath)
	}

	inference.InitEngine(verbose)

	modelConfig := inference.ModelConfig{
		ModelPath:  modelPath,
		ContextCtx: ctxLength,
		NumThreads: numThreads,
	}

	targetSchema := parseSchemaFlag(schemaRaw)
	var gbnfRules string
	if len(targetSchema) > 0 {
		gbnfRules = mask.GenerateJSONSchemaGBNF(targetSchema)
	}

	samplerConfig := inference.SamplerConfig{
		Temperature: temperature,
		TopK:        40,
		TopP:        0.95,
		Seed:        0,
		GrammarStr:  gbnfRules,
	}

	streamConfig := engine.StreamConfig{
		FlushTimeout: 50 * time.Millisecond,
		MaxLines:     1,
		MaxBytes:     1024,
	}

	runtime, err := inference.LoadModel(modelConfig)
	if err != nil {
		log.Fatalf("[!] Runtime initialization failure: %v\n", err)
	}
	defer runtime.Free()

	session := inference.NewSession(&runtime, 0)

	// Fixed: Structured context framing with positive and negative extraction anchors
	systemRules := []inference.Message{
		{
			Role:    "system",
			Content: "You are a precise security log parser. Extract the network source IP address into the 'ip' field, and the core event message or classification into the 'vector' field. If a field does not exist in the log line, output an empty string \"\" immediately.\n\nExample 1:\nInput: sshd[999]: Accepted publickey for root from 10.0.0.15 port 22 ssh2\nOutput: {\n  \"ip\": \"10.0.0.15\",\n  \"vector\": \"Accepted publickey for root\"\n}\n\nExample 2:\nInput: Cron jobs completed successfully\nOutput: {\n  \"ip\": \"\",\n  \"vector\": \"Cron jobs completed successfully\"\n}",
		},
	}
	basePrompt, _ := inference.FormatMessages(inference.TemplateChatML, systemRules)

	chunks := make(chan string, 64)
	go engine.ReadStream(os.Stdin, streamConfig, chunks)

	for chunk := range chunks {
		session.Reset()

		turnMessages := []inference.Message{
			{
				Role:    "user",
				Content: fmt.Sprintf("Input: %s\nOutput: ", chunk),
			},
		}

		formattedTurn, _ := inference.FormatMessages(inference.TemplateChatML, turnMessages)

		// Unify context tracking matching the exact pattern of the examples
		fullPrompt := basePrompt + formattedTurn + "{\n"
		fullTokens, err := inference.Tokenize(runtime.Model, fullPrompt, true)
		if err != nil {
			log.Fatalf("[!] Tokenization error: %v\n", err)
		}

		fmt.Print("{\n")
		os.Stdout.Sync()

		err = session.ExecuteTurn(samplerConfig, fullTokens, maxTokens, func(token string) {
			fmt.Print(token)
			os.Stdout.Sync()
		})
		if err != nil {
			log.Fatalf("\n[!] Stream turn evaluation failure: %v\n", err)
		}
		fmt.Println()
	}
}
