package main

import (
	"fmt"
	"log"
	"os"

	"glim/internal/inference"
)

func main() {
	fmt.Println("[*] Stabilizing core engine interfaces...")
	inference.InitEngine()

	config := inference.ModelConfig{
		ModelPath:  "/mnt/d/Code/glim/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
		ContextCtx: 2048,
		NumThreads: 4,
	}

	sampler := inference.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		Seed:        0, // 0 triggers automated cryptographic entropy sourcing
	}

	runtime, err := inference.LoadModel(config)
	if err != nil {
		log.Fatalf("[!] CGO Allocation crash: %v\n", err)
	}
	defer runtime.Free()

	// Initialize persistent state tracking across consecutive generation boundaries
	session := inference.NewSession(&runtime, 0)

	// --- Turn 1 ---
	history := []inference.Message{
		{Role: "system", Content: "You are a concise offensive security architecture expert."},
		{Role: "user", Content: "Name the primary API call monitored for standard LSASS parsing."},
	}

	prompt1, _ := inference.FormatMessages(inference.TemplateChatML, history)
	tokens1, _ := inference.Tokenize(runtime.Model, prompt1, true)

	fmt.Println("\n[Turn 1] Assistant:")
	err = session.ExecuteTurn(sampler, tokens1, 64, func(token string) {
		fmt.Print(token)
		os.Stdout.Sync()
	})
	if err != nil {
		log.Fatalf("\n[!] Turn 1 execution crash: %v\n", err)
	}
	fmt.Println()

	// --- Turn 2 (Incremental context execution - NO full cache clear) ---
	followUp := []inference.Message{
		{Role: "user", Content: "Give me an alternate user-mode alternative that bypasses it."},
	}

	// We only tokenize the delta because the sequence state is retained in the KV matrix
	prompt2, _ := inference.FormatMessages(inference.TemplateChatML, followUp)
	tokens2, _ := inference.Tokenize(runtime.Model, prompt2, false) // False: do not append standalone BOS token markers

	fmt.Println("\n[Turn 2] Assistant:")
	err = session.ExecuteTurn(sampler, tokens2, 128, func(token string) {
		fmt.Print(token)
		os.Stdout.Sync()
	})
	if err != nil {
		log.Fatalf("\n[!] Turn 2 execution crash: %v\n", err)
	}
	fmt.Println("\n\n--- Session Execution Phase Cleanly Complete ---")
}
