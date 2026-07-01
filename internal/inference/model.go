package inference

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../bin -lllama -lggml
#include <stdlib.h>
#include <stdbool.h>
#include "llama.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type ModelConfig struct {
	ModelPath  string
	NumThreads int
	ContextCtx int
}

type LlamaRuntime struct {
	Model *C.struct_llama_model
	Ctx   *C.struct_llama_context
}

func InitEngine() {
	C.llama_backend_init()

	cPath := C.CString("/mnt/d/Code/glim/bin")
	defer C.free(unsafe.Pointer(cPath))

	C.ggml_backend_load_all_from_path(cPath)
}

func Close(rt LlamaRuntime) {
	if rt.Ctx != nil {
		C.llama_free(rt.Ctx)
	}
	if rt.Model != nil {
		C.llama_model_free(rt.Model)
	}
	C.llama_backend_free()
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
		return rt, errors.New("failed to allocate dynamic context matrix via CGO layer")
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
