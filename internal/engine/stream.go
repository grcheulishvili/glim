package engine

import (
	"bufio"
	"io"
	"strings"
	"time"
)

type StreamConfig struct {
	FlushTimeout time.Duration
	MaxLines     int
	MaxBytes     int
}

// ReadStream continuously monitors the input source and passes formatted text blocks to the output channel.
func ReadStream(source io.Reader, config StreamConfig, outChan chan<- string) {
	defer close(outChan)
	scanner := bufio.NewScanner(source)

	var buffer strings.Builder
	var lineCount int

	lineChan := make(chan string)
	errChan := make(chan error, 1)

	// Background worker isolates blocking file/pipe read subsystems from scheduling loops
	go func() {
		for scanner.Scan() {
			lineChan <- scanner.Text()
		}
		if err := scanner.Err(); err != nil {
			errChan <- err
		}
		close(lineChan)
	}()

	timer := time.NewTimer(config.FlushTimeout)
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}

	flush := func() {
		if buffer.Len() > 0 {
			outChan <- buffer.String()
			buffer.Reset()
			lineCount = 0
		}
	}

	for {
		select {
		case line, ok := <-lineChan:
			if !ok {
				flush()
				return
			}

			if buffer.Len() > 0 {
				buffer.WriteString("\n")
			}
			buffer.WriteString(line)
			lineCount++

			if lineCount >= config.MaxLines || buffer.Len() >= config.MaxBytes {
				flush()
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
			} else {
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(config.FlushTimeout)
			}

		case <-timer.C:
			flush()

		case err := <-errChan:
			if err != nil {
				flush()
				return
			}
		}
	}
}