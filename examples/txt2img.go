package main

import (
	"fmt"
	stablediffusion "github.com/orangelang/stable-diffusion-go"
)

func main() {
	fmt.Println("Stable Diffusion Go - Text to Image Example")
	fmt.Println("===============================================")

	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "D:\\hf-mirror\\Z-Image-Turbo-GGUF\\z_image_turbo-Q4_K_M.gguf",
		LLMPath:            "D:\\hf-mirror\\Z-Image-Turbo-GGUF\\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
		VAEPath:            "D:\\hf-mirror\\Z-Image-Turbo-GGUF\\diffusion_pytorch_model.safetensors",
		DiffusionFlashAttn: true,
		OffloadParamsToCPU: true,
	})

	if err != nil {
		fmt.Println("Failed to create stable diffusion instance:", err)
		return
	}
	defer sd.Free()

	err = sd.GenerateImage(&stablediffusion.ImgGenParams{
		Prompt:      "生成一张世外桃源的风景图片，有山有水，有花有草，蝴蝶飞舞，有鸟有鱼",
		Width:       512,
		Height:      512,
		SampleSteps: 10,
		CfgScale:    1.0,
	}, "output_demo.png")

	if err != nil {
		fmt.Println("Failed to generate image:", err)
		return
	}

	fmt.Println("Image generated successfully!")

}
