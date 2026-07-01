package main

import (
	"fmt"
	"os"
	"path/filepath"

	"glim/internal/inference"
)

func main() {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	modelPath := filepath.Join(wd, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

	inference.InitEngine()

	config := inference.ModelConfig{
		ModelPath:  modelPath,
		NumThreads: 4,
		ContextCtx: 2048,
	}

	fmt.Printf("[*] Compiling memory context allocations for: %s\n", modelPath)
	runtime, err := inference.LoadModel(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[!] CGO Allocation crash: %v\n", err)
		os.Exit(1)
	}
	defer inference.Close(runtime)

	fmt.Println("[+] CGO context matrix stable. Testing tokenizer layer...")

	testString := "System anomaly detected: unexpected port 443 execution."
	tokens, err := inference.Tokenize(runtime.Model, testString, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[!] Token processing failure: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[+] Tokenization Success: %v\n", tokens)
}
