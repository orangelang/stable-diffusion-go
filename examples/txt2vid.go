package main

import (
	"fmt"
	stablediffusion "github.com/orangelang/stable-diffusion-go"
)

func main() {
	fmt.Println("Stable Diffusion Go - Text to Video Example")
	fmt.Println("================================================")

	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "D:\\hf-mirror\\wan2.1\\wan2.1_t2v_1.3B_bf16.safetensors",
		T5XXLPath:          "D:\\hf-mirror\\wan2.1\\umt5-xxl-encoder-Q4_K_M.gguf",
		VAEPath:            "D:\\hf-mirror\\wan2.1\\wan_2.1_vae.safetensors",
		DiffusionFlashAttn: true,
		KeepClipOnCPU:      true,
		OffloadParamsToCPU: true,
		NThreads:           4,
		FlowShift:          3.0,
	})

	if err != nil {
		fmt.Println("Failed to create stable diffusion instance:", err)
		return
	}
	defer sd.Free()

	err = sd.GenerateVideo(&stablediffusion.VidGenParams{
		Prompt:      "一个在长满桃花树下拍照的美女",
		Width:       300,
		Height:      300,
		SampleSteps: 40,
		VideoFrames: 33,
		CfgScale:    6.0,
	}, "./output.mp4")

	if err != nil {
		fmt.Println("Failed to generate video:", err)
		return
	}

	fmt.Println("Video generated successfully!")

}
