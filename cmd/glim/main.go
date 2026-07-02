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

	fmt.Printf("[*] Compiling memory context allocations for: %s\n", config.ModelPath)
	runtime, err := inference.LoadModel(config)
	if err != nil {
		log.Fatalf("[!] CGO Allocation crash: %v\n", err)
	}
	defer runtime.Free()

	// Verify target pipeline execution string
	prompt := "<|im_start|>user\nWrite a short shell script to check open ports.<|im_end|>\n<|im_start|>assistant\n"
	tokens, err := inference.Tokenize(runtime.Model, prompt, true)
	if err != nil {
		log.Fatalf("[!] Tokenizer step failure: %v\n", err)
	}

	fmt.Printf("[+] Processing Tokenization Array: %v\n", tokens)
	fmt.Println("\n--- Assistant Generation Stream ---")

	err = inference.DecodeStream(runtime, tokens, 128, func(token string) {
		fmt.Print(token)
		os.Stdout.Sync() // Flush standard out matrix continuously for real-time text visualization
	})
	if err != nil {
		log.Fatalf("\n[!] Loop evaluation crash: %v\n", err)
	}

	fmt.Println("\n\n--- Generation Cycle Cleanly Complete ---")
}
