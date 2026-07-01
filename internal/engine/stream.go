package engine

import (
	"bufio"
	"io"
	"time"
)

type Chunk struct {
	Lines []string
	Raw   string
}

type StreamConfig struct {
	MaxLines     int
	FlushTimeout time.Duration
}

func ProcessStream(reader io.Reader, config StreamConfig) <-chan Chunk {
	out := make(chan Chunk, 100)

	go func() {
		defer close(out)
		scanner := bufio.NewScanner(reader)

		var currentLines []string
		ticker := time.NewTicker(config.FlushTimeout)
		defer ticker.Stop()

		lineChan := make(chan string)

		go func() {
			for scanner.Scan() {
				lineChan <- scanner.Text()
			}
			close(lineChan)
		}()

		for {
			select {
			case line, ok := <-lineChan:
				if !ok {
					if len(currentLines) > 0 {
						out <- packageChunk(currentLines)
					}
					return
				}

				currentLines = append(currentLines, line)

				if len(currentLines) >= config.MaxLines {
					out <- packageChunk(currentLines)
					currentLines = nil
					ticker.Reset(config.FlushTimeout)
				}

			case <-ticker.C:
				if len(currentLines) > 0 {
					out <- packageChunk(currentLines)
					currentLines = nil
				}
			}
		}
	}()

	return out
}

func packageChunk(lines []string) Chunk {
	var raw string
	for i, line := range lines {
		if i > 0 {
			raw += "\n"
		}
		raw += line
	}
	return Chunk{
		Lines: lines,
		Raw:   raw,
	}
}
