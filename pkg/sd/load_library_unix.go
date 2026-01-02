//go:build darwin || linux

package sd

import (
	"github.com/ebitengine/purego"
)

// Open dynamic library function - Unix platforms (macOS/Linux)
func openLibrary(name string) (uintptr, error) {
	// Unix systems use purego.Dlopen to load dynamic libraries
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

// Close dynamic library function - Unix platforms (macOS/Linux)
func closeLibrary(handle uintptr) error {
	// Unix systems use purego.Dlclose to release dynamic libraries
	return purego.Dlclose(handle)
}
