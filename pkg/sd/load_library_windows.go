//go:build windows

package sd

import (
	"golang.org/x/sys/windows"
)

// Open dynamic library function - Windows platform
func openLibrary(name string) (uintptr, error) {
	// Windows uses windows.LoadLibrary to load dynamic libraries
	handle, err := windows.LoadLibrary(name)
	return uintptr(handle), err
}

// Close dynamic library function - Windows platform
func closeLibrary(handle uintptr) error {
	// Windows uses windows.FreeLibrary to release dynamic libraries
	return windows.FreeLibrary(windows.Handle(handle))
}
