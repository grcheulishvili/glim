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
		fmt.Fprintf(os.Stderr, "<<>> Error getting working directory: %v\n", err)
		os.Exit(1)
	}

	libPath := filepath.Join(wd, "bin", "llama.dll")

	fmt.Printf("<<>> Loading engine runtime matrix from: %s\n", libPath)

	err = inference.InitEngine(libPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "<<>> Engine bootstrap failed: %v\n", err)
		os.Exit(1)
	}
	defer inference.Close(0)

	fmt.Println("<<>> Local inference core linked successfully. Ready to bind model tensors.")
}
