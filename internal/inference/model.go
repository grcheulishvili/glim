package inference

import (
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
)

type ModelConfig struct {
	ModelPath  string
	NumThreads int
}

type LlamaEngine struct {
	libHandle         uintptr
	llamaBackendInit  func()
	llamaBackendFree  func()
	llamaInitFromFile func(path string, params uintptr) uintptr
	llamaFree         func(model uintptr)
}

var engine LlamaEngine

func InitEngine(libPath string) error {
	var handle uintptr

	if runtime.GOOS == "windows" {
		binDir := filepath.Dir(libPath)

		kernel32 := syscall.NewLazyDLL("kernel32.dll")
		setDllDir := kernel32.NewProc("SetDllDirectoryW")

		if setDllDir.Find() == nil {
			utf16Dir, err := syscall.UTF16PtrFromString(binDir)
			if err == nil {
				_, _, _ = setDllDir.Call(uintptr(unsafe.Pointer(utf16Dir)))
			}
		}

		h, winErr := syscall.LoadLibrary(libPath)
		if winErr != nil {
			return fmt.Errorf("failed to load windows shared library (.dll): %w", winErr)
		}
		handle = uintptr(h)

		if setDllDir.Find() == nil {
			_, _, _ = setDllDir.Call(0)
		}
	} else {
		return errors.New("platform loading configuration not compiled for this environment target")
	}

	engine.libHandle = handle

	purego.RegisterLibFunc(&engine.llamaBackendInit, handle, "llama_backend_init")
	purego.RegisterLibFunc(&engine.llamaBackendFree, handle, "llama_backend_free")
	purego.RegisterLibFunc(&engine.llamaInitFromFile, handle, "llama_model_load_from_file")
	purego.RegisterLibFunc(&engine.llamaFree, handle, "llama_model_free")

	if engine.llamaBackendInit == nil || engine.llamaInitFromFile == nil {
		return errors.New("required llama.cpp structural symbols missing from library binary")
	}

	engine.llamaBackendInit()
	return nil
}

func Close(modelPtr uintptr) {
	if modelPtr != 0 && engine.llamaFree != nil {
		engine.llamaFree(modelPtr)
	}
	if engine.llamaBackendFree != nil {
		engine.llamaBackendFree()
	}
	if engine.libHandle != 0 {
		if runtime.GOOS == "windows" {
			syscall.FreeLibrary(syscall.Handle(engine.libHandle))
		}
	}
}

func LoadModel(config ModelConfig) (uintptr, error) {
	if engine.llamaInitFromFile == nil {
		return 0, errors.New("engine layer not initialized; call InitEngine first")
	}

	modelPtr := engine.llamaInitFromFile(config.ModelPath, 0)
	if modelPtr == 0 {
		return 0, fmt.Errorf("failed to load model from path: %s", config.ModelPath)
	}

	return modelPtr, nil
}

func GetDefaultLibPath() string {
	if runtime.GOOS == "windows" {
		return "bin/llama.dll"
	}
	return "./libllama.so"
}
