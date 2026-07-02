package inference

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../bin -lllama -lggml
#include <stdlib.h>
#include <stdbool.h>
#include "llama.h"

// Helper to safely populate the sequence matrices inside C memory space
static void glim_batch_set_token(struct llama_batch *batch, int32_t i, llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits) {
    batch->token[i]    = token;
    batch->pos[i]      = pos;
    batch->n_seq_id[i] = 1;
    batch->seq_id[i][0] = seq_id;
    batch->logits[i]   = logits;
}

// Inline argmax sampler to isolate pipeline testing from sampler chain allocations
static llama_token glim_sample_greedy(struct llama_context *ctx, int32_t idx, int32_t n_vocab) {
    float *logits = llama_get_logits_ith(ctx, idx);
    llama_token max_token = 0;
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_token = i;
        }
    }
    return max_token;
}

// Wrap token_to_piece execution tracking across the CGO boundary using vocabulary matrices
static int32_t glim_token_to_piece(const struct llama_vocab *vocab, llama_token token, char *buf, int32_t length) {
    return llama_token_to_piece(vocab, token, buf, length, 0, false);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type ModelConfig struct {
	ModelPath  string
	ContextCtx int
	NumThreads int
}

type LlamaRuntime struct {
	Model *C.struct_llama_model
	Ctx   *C.struct_llama_context
}

func InitEngine() {
	cPath := C.CString("/mnt/d/Code/glim/bin")
	defer C.free(unsafe.Pointer(cPath))
	C.llama_backend_init()
	C.ggml_backend_load_all_from_path(cPath)
}

func LoadModel(config ModelConfig) (LlamaRuntime, error) {
	var rt LlamaRuntime

	cPath := C.CString(config.ModelPath)
	defer C.free(unsafe.Pointer(cPath))

	mParams := C.llama_model_default_params()
	rt.Model = C.llama_model_load_from_file(cPath, mParams)
	if rt.Model == nil {
		return rt, fmt.Errorf("failed to load model from path: %s", config.ModelPath)
	}

	cParams := C.llama_context_default_params()
	cParams.n_ctx = C.uint32_t(config.ContextCtx)
	cParams.n_threads = C.int32_t(config.NumThreads)

	rt.Ctx = C.llama_init_from_model(rt.Model, cParams)
	if rt.Ctx == nil {
		C.llama_model_free(rt.Model)
		return rt, errors.New("failed to allocate dynamic context matrix")
	}

	return rt, nil
}

func Tokenize(model *C.struct_llama_model, input string, addSpecial bool) ([]int32, error) {
	if model == nil {
		return nil, errors.New("provided model reference pointer is nil")
	}

	vocab := C.llama_model_get_vocab(model)
	if vocab == nil {
		return nil, errors.New("failed to retrieve vocabulary pointer matrix from model")
	}

	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	maxTokens := C.int(len(input) + 4)
	tokenBuffer := make([]C.llama_token, maxTokens)

	var addSpec C.bool = false
	if addSpecial {
		addSpec = true
	}

	nTokens := C.llama_tokenize(
		vocab,
		cInput,
		C.int(len(input)),
		&tokenBuffer[0],
		maxTokens,
		addSpec,
		true,
	)

	if nTokens < 0 {
		return nil, errors.New("internal tokenizer processing anomaly")
	}

	goTokens := make([]int32, nTokens)
	for i := 0; i < int(nTokens); i++ {
		goTokens[i] = int32(tokenBuffer[i])
	}

	return goTokens, nil
}

func (lr *LlamaRuntime) Free() {
	if lr.Ctx != nil {
		C.llama_free(lr.Ctx)
	}
	if lr.Model != nil {
		C.llama_model_free(lr.Model)
	}
}

// DecodeStream processes the initial batch and loops token evaluation sequentially
func DecodeStream(rt LlamaRuntime, promptTokens []int32, maxOutputTokens int, onToken func(string)) error {
	if rt.Model == nil || rt.Ctx == nil {
		return errors.New("uninitialized runtime execution matrix")
	}

	vocab := C.llama_model_get_vocab(rt.Model)
	if vocab == nil {
		return errors.New("failed to target vocabulary context for generation tracking")
	}

	// Fixed: Shifted from llama_n_vocab(model) to modern vocabulary token count interface
	nVocab := C.llama_vocab_n_tokens(vocab)

	batchSize := len(promptTokens)
	if maxOutputTokens > batchSize {
		batchSize = maxOutputTokens
	}

	batch := C.llama_batch_init(C.int32_t(batchSize+4), 0, 1)
	defer C.llama_batch_free(batch)

	for i, token := range promptTokens {
		isLast := i == len(promptTokens)-1
		C.glim_batch_set_token(&batch, C.int32_t(i), C.llama_token(token), C.llama_pos(i), 0, C.bool(isLast))
	}
	batch.n_tokens = C.int32_t(len(promptTokens))

	C.llama_memory_seq_rm(C.llama_get_memory(rt.Ctx), -1, -1, -1)

	var currentPos int32 = 0

	if C.llama_decode(rt.Ctx, batch) != 0 {
		return errors.New("failed to execute primary decode prompt batch evaluation")
	}

	currentPos += int32(batch.n_tokens)

	tokenBuf := make([]byte, 256)
	lastToken := C.glim_sample_greedy(rt.Ctx, batch.n_tokens-1, nVocab)

	for i := 0; i < maxOutputTokens; i++ {
		// Fixed: Shifted from llama_token_is_eog(model) to modern vocabulary constraint check
		if C.llama_vocab_is_eog(vocab, lastToken) {
			break
		}

		cBuf := (*C.char)(unsafe.Pointer(&tokenBuf[0]))
		nBytes := C.glim_token_to_piece(vocab, lastToken, cBuf, C.int32_t(len(tokenBuf)))
		if nBytes > 0 {
			onToken(string(tokenBuf[:nBytes]))
		}

		batch.n_tokens = 0
		C.glim_batch_set_token(&batch, 0, lastToken, C.llama_pos(currentPos), 0, true)
		batch.n_tokens = 1
		currentPos++

		if C.llama_decode(rt.Ctx, batch) != 0 {
			return errors.New("failed to evaluate sequence tracking decode matrix step")
		}

		lastToken = C.glim_sample_greedy(rt.Ctx, 0, nVocab)
	}

	return nil
}
