package inference

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../bin -Wl,-Bstatic -lllama -lggml -lggml-cpu -lggml-base -Wl,-Bdynamic -lm -lstdc++ -lgomp -lpthread -ldl
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "llama.h"

static bool glim_verbose_enabled = false;

static void glim_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (glim_verbose_enabled) {
        fprintf(stderr, "%s", text);
        fflush(stderr);
    }
}

static void glim_setup_logging(bool verbose) {
    glim_verbose_enabled = verbose;
    llama_log_set(glim_log_callback, NULL);
}

static void glim_batch_set_token(struct llama_batch *batch, int32_t i, llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits) {
    batch->token[i]    = token;
    batch->pos[i]      = pos;
    batch->n_seq_id[i] = 1;
    batch->seq_id[i][0] = seq_id;
    batch->logits[i]   = logits;
}

static int32_t glim_token_to_piece(const struct llama_vocab *vocab, llama_token token, char *buf, int32_t length) {
    return llama_token_to_piece(vocab, token, buf, length, 0, false);
}
*/
import "C"

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

type ChatTemplate string

const (
	TemplateChatML ChatTemplate = "chatml"
)

type Message struct {
	Role    string
	Content string
}

type SamplerConfig struct {
	Temperature float32
	TopK        int32
	TopP        float32
	Seed        uint32
	GrammarStr  string
}

type ModelConfig struct {
	ModelPath  string
	ContextCtx int
	NumThreads int
}

type LlamaRuntime struct {
	Model *C.struct_llama_model
	Ctx   *C.struct_llama_context
	Vocab *C.struct_llama_vocab
}

type ContextSession struct {
	Runtime    *LlamaRuntime
	SequenceID int32
	CurrentPos int32
}

func InitEngine(verbose bool) {
	cPath := C.CString("/mnt/d/Code/glim/bin")
	defer C.free(unsafe.Pointer(cPath))

	C.glim_setup_logging(C.bool(verbose))
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

	rt.Vocab = C.llama_model_get_vocab(rt.Model)
	if rt.Vocab == nil {
		C.llama_model_free(rt.Model)
		return rt, errors.New("failed to retrieve model vocabulary pointer map")
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

func NewSession(rt *LlamaRuntime, seqID int32) *ContextSession {
	// Fixed: Reverted to the native llama_memory_seq_rm API using the context memory pointer
	C.llama_memory_seq_rm(C.llama_get_memory(rt.Ctx), C.llama_seq_id(seqID), 0, -1)
	return &ContextSession{
		Runtime:    rt,
		SequenceID: seqID,
		CurrentPos: 0,
	}
}

func (cs *ContextSession) Reset() {
	if cs.Runtime.Ctx == nil {
		return
	}
	// Fixed: Reverted to the native llama_memory_seq_rm API using the context memory pointer
	C.llama_memory_seq_rm(
		C.llama_get_memory(cs.Runtime.Ctx),
		C.llama_seq_id(cs.SequenceID),
		0,
		-1,
	)
	cs.CurrentPos = 0
}

func FormatMessages(template ChatTemplate, messages []Message) (string, error) {
	var sb strings.Builder
	for _, msg := range messages {
		switch template {
		case TemplateChatML:
			sb.WriteString(fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", msg.Role, msg.Content))
		default:
			return "", fmt.Errorf("unsupported system chat template: %s", template)
		}
	}
	if len(messages) > 0 && messages[len(messages)-1].Role == "user" {
		sb.WriteString("<|im_start|>assistant\n")
	}
	return sb.String(), nil
}

func Tokenize(model *C.struct_llama_model, input string, addSpecial bool) ([]int32, error) {
	vocab := C.llama_model_get_vocab(model)
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

func (cs *ContextSession) ExecuteTurn(smpl SamplerConfig, newTokens []int32, maxOutputTokens int, onToken func(string)) error {
	if len(newTokens) == 0 {
		return errors.New("empty processing token slice passed to generation layer")
	}

	batchSize := len(newTokens)
	if maxOutputTokens > batchSize {
		batchSize = maxOutputTokens
	}

	batch := C.llama_batch_init(C.int32_t(batchSize+4), 0, 1)
	defer C.llama_batch_free(batch)

	for i, token := range newTokens {
		isLast := i == len(newTokens)-1
		C.glim_batch_set_token(
			&batch,
			C.int32_t(i),
			C.llama_token(token),
			C.llama_pos(cs.CurrentPos+int32(i)),
			C.llama_seq_id(cs.SequenceID),
			C.bool(isLast),
		)
	}
	batch.n_tokens = C.int32_t(len(newTokens))

	sparams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(sparams)
	defer C.llama_sampler_free(chain)

	if smpl.GrammarStr != "" {
		cGrammar := C.CString(smpl.GrammarStr)
		cRoot := C.CString("root")
		defer C.free(unsafe.Pointer(cGrammar))
		defer C.free(unsafe.Pointer(cRoot))

		grammarSampler := C.llama_sampler_init_grammar(cs.Runtime.Vocab, cGrammar, cRoot)
		if grammarSampler != nil {
			C.llama_sampler_chain_add(chain, grammarSampler)
		}
	}

	if smpl.TopK > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(C.int32_t(smpl.TopK)))
	}
	if smpl.TopP > 0.0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(C.float(smpl.TopP), 1))
	}
	if smpl.Temperature > 0.0 {
		seed := smpl.Seed
		if seed == 0 {
			var b [4]byte
			_, _ = rand.Read(b[:])
			seed = binary.LittleEndian.Uint32(b[:])
		}
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(smpl.Temperature)))
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(seed)))
	} else {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
	}

	if C.llama_decode(cs.Runtime.Ctx, batch) != 0 {
		return errors.New("failed to execute incremental matrix sequence evaluation step")
	}
	cs.CurrentPos += int32(batch.n_tokens)

	tokenBuf := make([]byte, 256)
	lastToken := C.llama_sampler_sample(chain, cs.Runtime.Ctx, batch.n_tokens-1)

	for i := 0; i < maxOutputTokens; i++ {
		if C.llama_vocab_is_eog(cs.Runtime.Vocab, lastToken) {
			break
		}

		cBuf := (*C.char)(unsafe.Pointer(&tokenBuf[0]))
		nBytes := C.glim_token_to_piece(cs.Runtime.Vocab, lastToken, cBuf, C.int32_t(len(tokenBuf)))
		if nBytes > 0 {
			onToken(string(tokenBuf[:nBytes]))
		}

		batch.n_tokens = 0
		C.glim_batch_set_token(
			&batch,
			0,
			lastToken,
			C.llama_pos(cs.CurrentPos),
			C.llama_seq_id(cs.SequenceID),
			true,
		)
		batch.n_tokens = 1
		cs.CurrentPos++

		if C.llama_decode(cs.Runtime.Ctx, batch) != 0 {
			return errors.New("failed to evaluate generation continuation token")
		}

		lastToken = C.llama_sampler_sample(chain, cs.Runtime.Ctx, 0)
	}

	return nil
}
