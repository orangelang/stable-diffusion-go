package sd

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Define enum types

type RngType int32

const (
	DefaultRNG RngType = iota
	CUDARNG
	CPURNG
	RNGTypeCount
)

type SampleMethod int32

const (
	EulerSampleMethod SampleMethod = iota
	EulerASampleMethod
	HeunSampleMethod
	DPM2SampleMethod
	DPMPP2SASampleMethod
	DPMPP2MSampleMethod
	DPMPP2Mv2SampleMethod
	IPNDMSampleMethod
	IPNDMSampleMethodV
	LCMSampleMethod
	DDIMTrailingSampleMethod
	TCDSampleMethod
	SampleMethodCount
)

type Scheduler int32

const (
	DiscreteScheduler Scheduler = iota
	KarrasScheduler
	ExponentialScheduler
	AYSScheduler
	GITScheduler
	SGMUniformScheduler
	SimpleScheduler
	SmoothstepScheduler
	KLOptimalScheduler
	LCMScheduler
	SchedulerCount
)

type Prediction int32

const (
	EPSPred Prediction = iota
	VPred
	EDMVPred
	FlowPred
	FluxFlowPred
	Flux2FlowPred
	PredictionCount
)

type SDType int32

const (
	SDTypeF32 SDType = iota
	SDTypeF16
	SDTypeQ4_0
	SDTypeQ4_1
	// SDTypeQ4_2 = 4, support has been removed
	// SDTypeQ4_3 = 5, support has been removed
	SDTypeQ5_0    = 6
	SDTypeQ5_1    = 7
	SDTypeQ8_0    = 8
	SDTypeQ8_1    = 9
	SDTypeQ2_K    = 10
	SDTypeQ3_K    = 11
	SDTypeQ4_K    = 12
	SDTypeQ5_K    = 13
	SDTypeQ6_K    = 14
	SDTypeQ8_K    = 15
	SDTypeIQ2_XXS = 16
	SDTypeIQ2_XS  = 17
	SDTypeIQ3_XXS = 18
	SDTypeIQ1_S   = 19
	SDTypeIQ4_NL  = 20
	SDTypeIQ3_S   = 21
	SDTypeIQ2_S   = 22
	SDTypeIQ4_XS  = 23
	SDTypeI8      = 24
	SDTypeI16     = 25
	SDTypeI32     = 26
	SDTypeI64     = 27
	SDTypeF64     = 28
	SDTypeIQ1_M   = 29
	SDTypeBF16    = 30
	// SDTypeQ4_0_4_4 = 31, support has been removed from gguf files
	// SDTypeQ4_0_4_8 = 32,
	// SDTypeQ4_0_8_8 = 33,
	SDTypeTQ1_0 = 34
	SDTypeTQ2_0 = 35
	// SDTypeIQ4_NL_4_4 = 36,
	// SDTypeIQ4_NL_4_8 = 37,
	// SDTypeIQ4_NL_8_8 = 38,
	SDTypeMXFP4 = 39
	SDTypeCount = 40
)

type SDLogLevel int32

const (
	SDLogDebug SDLogLevel = iota
	SDLogInfo
	SDLogWarn
	SDLogError
)

type Preview int32

const (
	PreviewNone Preview = iota
	PreviewProj
	PreviewTAE
	PreviewVAE
	PreviewCount
)

type LoraApplyMode int32

const (
	LoraApplyAuto LoraApplyMode = iota
	LoraApplyImmediately
	LoraApplyAtRuntime
	LoraApplyModeCount
)

type SDCacheMode int32

const (
	SDCacheDisabled SDCacheMode = iota
	SDCacheEasycache
	SDCacheUcache
	SDCacheDbcache
	SDCacheTaylorseer
	SDCacheCacheDit
)

// Define structs
type SDTilingParams struct {
	Enabled       bool
	TileSizeX     int32
	TileSizeY     int32
	TargetOverlap float32
	RelSizeX      float32
	RelSizeY      float32
}

type SDEmbedding struct {
	Name *uint8
	Path *uint8
}

type SDContextParams struct {
	ModelPath                   *uint8
	ClipLPath                   *uint8
	ClipGPath                   *uint8
	ClipVisionPath              *uint8
	T5XXLPath                   *uint8
	LLMPath                     *uint8
	LLMVisionPath               *uint8
	DiffusionModelPath          *uint8
	HighNoiseDiffusionModelPath *uint8
	VAEPath                     *uint8
	TAESDPath                   *uint8
	ControlNetPath              *uint8
	Embeddings                  *SDEmbedding
	EmbeddingCount              uint32
	PhotoMakerPath              *uint8
	TensorTypeRules             *uint8
	VAEDecodeOnly               bool
	FreeParamsImmediately       bool
	NThreads                    int32
	WType                       SDType
	RNGType                     RngType
	SamplerRNGType              RngType
	Prediction                  Prediction
	LoraApplyMode               LoraApplyMode
	OffloadParamsToCPU          bool
	EnableMmap                  bool
	KeepClipOnCPU               bool
	KeepControlNetOnCPU         bool
	KeepVAEOnCPU                bool
	DiffusionFlashAttn          bool
	TAEPreviewOnly              bool
	DiffusionConvDirect         bool
	VAEConvDirect               bool
	CircularX                   bool
	CircularY                   bool
	ForceSDXLVAConvScale        bool
	ChromaUseDitMask            bool
	ChromaUseT5Mask             bool
	ChromaT5MaskPad             int32
	QwenImageZeroCondT          bool
	FlowShift                   float32
}

type SDImage struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    *uint8
}

type SDSLGParams struct {
	Layers     *int32
	LayerCount uintptr
	LayerStart float32
	LayerEnd   float32
	Scale      float32
}

type SDGuidanceParams struct {
	TxtCfg            float32
	ImgCfg            float32
	DistilledGuidance float32
	SLG               SDSLGParams
}

type SDSampleParams struct {
	Guidance          SDGuidanceParams
	Scheduler         Scheduler
	SampleMethod      SampleMethod
	SampleSteps       int32
	Eta               float32
	ShiftedTimestep   int32
	CustomSigmas      *float32
	CustomSigmasCount int32
}

type SDPMParams struct {
	IDImages      *SDImage
	IDImagesCount int32
	IDEmbedPath   *uint8
	StyleStrength float32
}

type SDCacheParams struct {
	Mode                     SDCacheMode
	ReuseThreshold           float32
	StartPercent             float32
	EndPercent               float32
	ErrorDecayRate           float32
	UseRelativeThreshold     bool
	ResetErrorOnCompute      bool
	FnComputeBlocks          int32
	BnComputeBlocks          int32
	ResidualDiffThreshold    float32
	MaxWarmupSteps           int32
	MaxCachedSteps           int32
	MaxContinuousCachedSteps int32
	TaylorseerNDerivatives   int32
	TaylorseerSkipInterval   int32
	ScmMask                  *uint8
	ScmPolicyDynamic         bool
}

type SDLora struct {
	IsHighNoise bool
	Multiplier  float32
	Path        *uint8
}

type SDImgGenParams struct {
	Loras              *SDLora
	LoraCount          uint32
	Prompt             *uint8
	NegativePrompt     *uint8
	ClipSkip           int32
	InitImage          SDImage
	RefImages          *SDImage
	RefImagesCount     int32
	AutoResizeRefImage bool
	IncreaseRefIndex   bool
	MaskImage          SDImage
	Width              int32
	Height             int32
	SampleParams       SDSampleParams
	Strength           float32
	Seed               int64
	BatchCount         int32
	ControlImage       SDImage
	ControlStrength    float32
	PMParams           SDPMParams
	VAETilingParams    SDTilingParams
	Cache              SDCacheParams
}

type SDVidGenParams struct {
	Loras                 *SDLora
	LoraCount             uint32
	Prompt                *uint8
	NegativePrompt        *uint8
	ClipSkip              int32
	InitImage             SDImage
	EndImage              SDImage
	ControlFrames         *SDImage
	ControlFramesSize     int32
	Width                 int32
	Height                int32
	SampleParams          SDSampleParams
	HighNoiseSampleParams SDSampleParams
	MOEBoundary           float32
	Strength              float32
	Seed                  int64
	VideoFrames           int32
	VaceStrength          float32
	Cache                 SDCacheParams
}

// Define context types
type SDContext struct {
	ptr unsafe.Pointer
}

type UpscalerContext struct {
	ptr unsafe.Pointer
}

// Define callback function types
type SDLogCallback func(level SDLogLevel, text *uint8, data unsafe.Pointer)
type SDProgressCallback func(step int32, steps int32, time float32, data unsafe.Pointer)
type SDPreviewCallback func(step int32, frameCount int32, frames *SDImage, isNoisy bool, data unsafe.Pointer)

// Dynamic library function declarations
var (
	sdSetLogCallback         func(cb SDLogCallback, data unsafe.Pointer)
	sdSetProgressCallback    func(cb SDProgressCallback, data unsafe.Pointer)
	sdSetPreviewCallback     func(cb SDPreviewCallback, mode Preview, interval int32, denoised bool, noisy bool, data unsafe.Pointer)
	sdGetNumPhysicalCores    func() int32
	sdGetSystemInfo          func() *uint8
	sdTypeName               func(typ SDType) *uint8
	strToSDType              func(str *uint8) SDType
	sdRngTypeName            func(rngType RngType) *uint8
	strToRngType             func(str *uint8) RngType
	sdSampleMethodName       func(method SampleMethod) *uint8
	strToSampleMethod        func(str *uint8) SampleMethod
	sdSchedulerName          func(scheduler Scheduler) *uint8
	strToScheduler           func(str *uint8) Scheduler
	sdPredictionName         func(prediction Prediction) *uint8
	strToPrediction          func(str *uint8) Prediction
	sdPreviewName            func(preview Preview) *uint8
	strToPreview             func(str *uint8) Preview
	sdLoraApplyModeName      func(mode LoraApplyMode) *uint8
	strToLoraApplyMode       func(str *uint8) LoraApplyMode
	sdCacheParamsInit        func(params *SDCacheParams)
	sdContextParamsInit      func(params *SDContextParams)
	sdContextParamsToStr     func(params *SDContextParams) *uint8
	newSDContext             func(params *SDContextParams) unsafe.Pointer
	freeSDContext            func(ctx unsafe.Pointer)
	sdSampleParamsInit       func(params *SDSampleParams)
	sdSampleParamsToStr      func(params *SDSampleParams) *uint8
	sdGetDefaultSampleMethod func(ctx unsafe.Pointer) SampleMethod
	sdGetDefaultScheduler    func(ctx unsafe.Pointer, sampleMethod SampleMethod) Scheduler
	sdImgGenParamsInit       func(params *SDImgGenParams)
	sdImgGenParamsToStr      func(params *SDImgGenParams) *uint8
	generateImage            func(ctx unsafe.Pointer, params *SDImgGenParams) *SDImage
	sdVidGenParamsInit       func(params *SDVidGenParams)
	generateVideo            func(ctx unsafe.Pointer, params *SDVidGenParams, numFramesOut *int32) *SDImage
	newUpscalerContext       func(esrganPath *uint8, offloadParamsToCPU bool, direct bool, nThreads int32, tileSize int32) unsafe.Pointer
	freeUpscalerContext      func(ctx unsafe.Pointer)
	upscale                  func(ctx unsafe.Pointer, inputImage *SDImage, upscaleFactor uint32) *SDImage
	getUpscaleFactor         func(ctx unsafe.Pointer) int32
	convert                  func(inputPath *uint8, vaePath *uint8, outputPath *uint8, outputType SDType, tensorTypeRules *uint8, convertName bool) bool
	preprocessCanny          func(image *SDImage, highThreshold float32, lowThreshold float32, weak float32, strong float32, inverse bool) bool
	sdCommit                 func() *uint8
	sdVersion                func() *uint8
)

// Dynamic library handle
var libSD uintptr

// Load dynamic library
func init() {
	// Define dynamic library filename and path
	var libDir string
	var libPath string
	var err error
	var gpuName string

	switch runtime.GOOS {
	case "windows":
		// Determine the best library directory based on CPU architecture and GPU type
		if strings.ToLower(os.Getenv("SD_VK_DEVICE")) == "true" {
			var vulkanGPU string
			vulkanGPU, err = GetVulkanGPU()
			if err != nil || vulkanGPU == "" {
				fmt.Println("Warning: Failed to get Vulkan GPU")
				return
			}

			libPath = GetSDLibPath("vulkan/stable-diffusion.dll")
			libSD, err = openLibrary(libPath)

		} else {
			gpuName, err = GetGPUName()
			if err != nil {
				fmt.Println("Warning: Failed to get GPU name: " + err.Error())
				return
			}

			if gpuName == "NVIDIA" {
				libPath = GetSDLibPath("cuda12/stable-diffusion.dll")
				libSD, err = openLibrary(libPath)
			} else if gpuName == "AMD" {
				libPath = GetSDLibPath("rocm/stable-diffusion.dll")
				libSD, err = openLibrary(libPath)
			}
		}

		if err != nil {
			fmt.Println("Warning: Failed to load stable-diffusion library from path: " + libPath)
			fmt.Println("Trying to load from GPU accelerated library failed, trying to load from CPU library")
			libDir = GetCpuAVX()
			libPath = GetSDLibPath(libDir + "/stable-diffusion.dll")
			libSD, err = openLibrary(libPath)
		}

	case "darwin":
		libPath = GetSDLibPath("libstable-diffusion.dylib")
		libSD, err = openLibrary(libPath)
	default: // linux
		libPath = GetSDLibPath("libstable-diffusion.so")
		libSD, err = openLibrary(libPath)
	}

	if err != nil || libSD == 0 {
		fmt.Println("Warning: Failed to load stable-diffusion library from path: " + libPath)
		fmt.Println(err)
		return
	}

	fmt.Println("Loaded stable-diffusion library: " + libPath)

	// Bind functions
	purego.RegisterLibFunc(&sdSetLogCallback, libSD, "sd_set_log_callback")
	purego.RegisterLibFunc(&sdSetProgressCallback, libSD, "sd_set_progress_callback")
	purego.RegisterLibFunc(&sdSetPreviewCallback, libSD, "sd_set_preview_callback")
	purego.RegisterLibFunc(&sdGetNumPhysicalCores, libSD, "sd_get_num_physical_cores")
	purego.RegisterLibFunc(&sdGetSystemInfo, libSD, "sd_get_system_info")
	purego.RegisterLibFunc(&sdTypeName, libSD, "sd_type_name")
	purego.RegisterLibFunc(&strToSDType, libSD, "str_to_sd_type")
	purego.RegisterLibFunc(&sdRngTypeName, libSD, "sd_rng_type_name")
	purego.RegisterLibFunc(&strToRngType, libSD, "str_to_rng_type")
	purego.RegisterLibFunc(&sdSampleMethodName, libSD, "sd_sample_method_name")
	purego.RegisterLibFunc(&strToSampleMethod, libSD, "str_to_sample_method")
	purego.RegisterLibFunc(&sdSchedulerName, libSD, "sd_scheduler_name")
	purego.RegisterLibFunc(&strToScheduler, libSD, "str_to_scheduler")
	purego.RegisterLibFunc(&sdPredictionName, libSD, "sd_prediction_name")
	purego.RegisterLibFunc(&strToPrediction, libSD, "str_to_prediction")
	purego.RegisterLibFunc(&sdPreviewName, libSD, "sd_preview_name")
	purego.RegisterLibFunc(&strToPreview, libSD, "str_to_preview")
	purego.RegisterLibFunc(&sdLoraApplyModeName, libSD, "sd_lora_apply_mode_name")
	purego.RegisterLibFunc(&strToLoraApplyMode, libSD, "str_to_lora_apply_mode")
	purego.RegisterLibFunc(&sdCacheParamsInit, libSD, "sd_cache_params_init")
	purego.RegisterLibFunc(&sdContextParamsInit, libSD, "sd_ctx_params_init")
	purego.RegisterLibFunc(&sdContextParamsToStr, libSD, "sd_ctx_params_to_str")
	purego.RegisterLibFunc(&newSDContext, libSD, "new_sd_ctx")
	purego.RegisterLibFunc(&freeSDContext, libSD, "free_sd_ctx")
	purego.RegisterLibFunc(&sdSampleParamsInit, libSD, "sd_sample_params_init")
	purego.RegisterLibFunc(&sdSampleParamsToStr, libSD, "sd_sample_params_to_str")
	purego.RegisterLibFunc(&sdGetDefaultSampleMethod, libSD, "sd_get_default_sample_method")
	purego.RegisterLibFunc(&sdGetDefaultScheduler, libSD, "sd_get_default_scheduler")
	purego.RegisterLibFunc(&sdImgGenParamsInit, libSD, "sd_img_gen_params_init")
	purego.RegisterLibFunc(&sdImgGenParamsToStr, libSD, "sd_img_gen_params_to_str")
	purego.RegisterLibFunc(&generateImage, libSD, "generate_image")
	purego.RegisterLibFunc(&sdVidGenParamsInit, libSD, "sd_vid_gen_params_init")
	purego.RegisterLibFunc(&generateVideo, libSD, "generate_video")
	purego.RegisterLibFunc(&newUpscalerContext, libSD, "new_upscaler_ctx")
	purego.RegisterLibFunc(&freeUpscalerContext, libSD, "free_upscaler_ctx")
	purego.RegisterLibFunc(&upscale, libSD, "upscale")
	purego.RegisterLibFunc(&getUpscaleFactor, libSD, "get_upscale_factor")
	purego.RegisterLibFunc(&convert, libSD, "convert")
	purego.RegisterLibFunc(&preprocessCanny, libSD, "preprocess_canny")
	purego.RegisterLibFunc(&sdCommit, libSD, "sd_commit")
	purego.RegisterLibFunc(&sdVersion, libSD, "sd_version")

	fmt.Println("Register lib func finish !")
}

// Wrapper functions
type SDLogLevelType int32

type SDLogCallbackType func(level SDLogLevelType, text string, data interface{})

// SetLogCallback sets log callback
func SetLogCallback(cb SDLogCallbackType, data interface{}) {
	if cb == nil {
		sdSetLogCallback(nil, nil)
		return
	}

	// Create a closure to convert Go callback to C callback
	cCallback := func(level SDLogLevel, text *uint8, cData unsafe.Pointer) {
		cb(SDLogLevelType(level), CGoString(text), data)
	}

	sdSetLogCallback(cCallback, nil)
}

// SetProgressCallback sets progress callback
func SetProgressCallback(cb func(step int, steps int, time float32, data interface{}), data interface{}) {
	if cb == nil {
		sdSetProgressCallback(nil, nil)
		return
	}

	cCallback := func(step int32, steps int32, time float32, cData unsafe.Pointer) {
		cb(int(step), int(steps), time, data)
	}

	sdSetProgressCallback(cCallback, nil)
}

// SetPreviewCallback sets preview callback
func SetPreviewCallback(cb func(step int, frameCount int, frames []SDImage, isNoisy bool, data interface{}), mode Preview, interval int, denoised bool, noisy bool, data interface{}) {
	if cb == nil {
		sdSetPreviewCallback(nil, mode, int32(interval), denoised, noisy, nil)
		return
	}

	cCallback := func(step int32, frameCount int32, cFrames *SDImage, isNoisy bool, cData unsafe.Pointer) {
		// Convert C pointer to Go slice
		frames := make([]SDImage, frameCount)
		for i := range frames {
			frames[i] = *(*SDImage)(unsafe.Add(unsafe.Pointer(cFrames), uintptr(i)*unsafe.Sizeof(SDImage{})))
		}
		cb(int(step), int(frameCount), frames, isNoisy, data)
	}

	sdSetPreviewCallback(cCallback, mode, int32(interval), denoised, noisy, nil)
}

// GetNumPhysicalCores gets the number of physical cores
func GetNumPhysicalCores() int {
	return int(sdGetNumPhysicalCores())
}

// GetSystemInfo gets system information
func GetSystemInfo() string {
	return CGoString(sdGetSystemInfo())
}

// SDTypeName gets SD type name
func SDTypeName(typ SDType) string {
	return CGoString(sdTypeName(typ))
}

// StrToSDType converts string to SD type
func StrToSDType(str string) SDType {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToSDType(cStr)
}

// RNGTypeName gets RNG type name
func RNGTypeName(rngType RngType) string {
	return CGoString(sdRngTypeName(rngType))
}

// StrToRNGType converts string to RNG type
func StrToRNGType(str string) RngType {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToRngType(cStr)
}

// SampleMethodName gets sample method name
func SampleMethodName(method SampleMethod) string {
	return CGoString(sdSampleMethodName(method))
}

// StrToSampleMethod converts string to sample method
func StrToSampleMethod(str string) SampleMethod {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToSampleMethod(cStr)
}

// SchedulerName gets scheduler name
func SchedulerName(scheduler Scheduler) string {
	return CGoString(sdSchedulerName(scheduler))
}

// StrToScheduler converts string to scheduler
func StrToScheduler(str string) Scheduler {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToScheduler(cStr)
}

// PredictionName gets prediction type name
func PredictionName(prediction Prediction) string {
	return CGoString(sdPredictionName(prediction))
}

// StrToPrediction converts string to prediction type
func StrToPrediction(str string) Prediction {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToPrediction(cStr)
}

// PreviewName gets preview type name
func PreviewName(preview Preview) string {
	return CGoString(sdPreviewName(preview))
}

// StrToPreview converts string to preview type
func StrToPreview(str string) Preview {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToPreview(cStr)
}

// LoraApplyModeName gets LoRA apply mode name
func LoraApplyModeName(mode LoraApplyMode) string {
	return CGoString(sdLoraApplyModeName(mode))
}

// StrToLoraApplyMode converts string to LoRA apply mode
func StrToLoraApplyMode(str string) LoraApplyMode {
	cStr := CString(str)
	defer FreeCString(cStr)
	return strToLoraApplyMode(cStr)
}

// CacheParamsInit initializes cache parameters
func CacheParamsInit(params *SDCacheParams) {
	sdCacheParamsInit(params)
}

// ContextParamsInit initializes context parameters
func ContextParamsInit(params *SDContextParams) {
	sdContextParamsInit(params)
}

// ContextParamsToStr converts context parameters to string
func ContextParamsToStr(params *SDContextParams) string {
	return CGoString(sdContextParamsToStr(params))
}

// NewContext creates a new context
func NewContext(params *SDContextParams) *SDContext {
	ptr := newSDContext(params)
	return &SDContext{ptr: ptr}
}

// FreeContext frees context
func (ctx *SDContext) Free() {
	if ctx.ptr != nil {
		freeSDContext(ctx.ptr)
		ctx.ptr = nil
	}
}

// SampleParamsInit initializes sample parameters
func SampleParamsInit(params *SDSampleParams) {
	sdSampleParamsInit(params)
}

// SampleParamsToStr converts sample parameters to string
func SampleParamsToStr(params *SDSampleParams) string {
	return CGoString(sdSampleParamsToStr(params))
}

// GetDefaultSampleMethod gets default sample method
func (ctx *SDContext) GetDefaultSampleMethod() SampleMethod {
	return sdGetDefaultSampleMethod(ctx.ptr)
}

// GetDefaultScheduler gets default scheduler
func (ctx *SDContext) GetDefaultScheduler(sampleMethod SampleMethod) Scheduler {
	return sdGetDefaultScheduler(ctx.ptr, sampleMethod)
}

// ImgGenParamsInit initializes image generation parameters
func ImgGenParamsInit(params *SDImgGenParams) {
	sdImgGenParamsInit(params)
}

// ImgGenParamsToStr converts image generation parameters to string
func ImgGenParamsToStr(params *SDImgGenParams) string {
	return CGoString(sdImgGenParamsToStr(params))
}

// GenerateImage generates image
func (ctx *SDContext) GenerateImage(params *SDImgGenParams) *SDImage {
	return generateImage(ctx.ptr, params)
}

// VidGenParamsInit initializes video generation parameters
func VidGenParamsInit(params *SDVidGenParams) {
	sdVidGenParamsInit(params)
}

// GenerateVideo generates video
func (ctx *SDContext) GenerateVideo(params *SDVidGenParams) ([]SDImage, int) {
	var numFrames int32
	framesPtr := generateVideo(ctx.ptr, params, &numFrames)
	if framesPtr == nil {
		return nil, 0
	}

	frames := make([]SDImage, numFrames)
	for i := range frames {
		frames[i] = *(*SDImage)(unsafe.Add(unsafe.Pointer(framesPtr), uintptr(i)*unsafe.Sizeof(SDImage{})))
	}

	return frames, int(numFrames)
}

// NewUpscalerContext creates a new upscaler context
func NewUpscalerContext(esrganPath string, offloadParamsToCPU bool, direct bool, nThreads int, tileSize int) *UpscalerContext {
	cPath := CString(esrganPath)
	defer FreeCString(cPath)

	ptr := newUpscalerContext(cPath, offloadParamsToCPU, direct, int32(nThreads), int32(tileSize))
	return &UpscalerContext{ptr: ptr}
}

// FreeUpscalerContext frees upscaler context
func (ctx *UpscalerContext) Free() {
	if ctx.ptr != nil {
		freeUpscalerContext(ctx.ptr)
		ctx.ptr = nil
	}
}

// Upscale upscales image
func (ctx *UpscalerContext) Upscale(inputImage SDImage, upscaleFactor uint32) SDImage {
	return *upscale(ctx.ptr, &inputImage, upscaleFactor)
}

// GetUpscaleFactor gets upscale factor
func (ctx *UpscalerContext) GetUpscaleFactor() int {
	return int(getUpscaleFactor(ctx.ptr))
}

// Convert converts model
func Convert(inputPath, vaePath, outputPath string, outputType SDType, tensorTypeRules string, convertName bool) bool {
	cInputPath := CString(inputPath)
	cVaePath := CString(vaePath)
	cOutputPath := CString(outputPath)
	cTensorTypeRules := CString(tensorTypeRules)

	defer func() {
		FreeCString(cInputPath)
		FreeCString(cVaePath)
		FreeCString(cOutputPath)
		FreeCString(cTensorTypeRules)
	}()

	return convert(cInputPath, cVaePath, cOutputPath, outputType, cTensorTypeRules, convertName)
}

// PreprocessCanny preprocesses Canny edge detection
func PreprocessCanny(image SDImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool {
	return preprocessCanny(&image, highThreshold, lowThreshold, weak, strong, inverse)
}

// Commit gets commit information
func Commit() string {
	return CGoString(sdCommit())
}

// Version gets version information
func Version() string {
	return CGoString(sdVersion())
}

// Helper function: Convert C string to Go string
func CGoString(cStr *uint8) string {
	if cStr == nil {
		return ""
	}

	// Calculate string length
	var len int
	for p := cStr; *p != 0; p = (*uint8)(unsafe.Add(unsafe.Pointer(p), 1)) {
		len++
	}

	// Convert to Go string
	return string(unsafe.Slice(cStr, len))
}

// Helper function: Convert Go string to C string
func CString(str string) *uint8 {
	if str == "" {
		return nil
	}

	// Allocate memory, including the NULL terminator
	buf := make([]uint8, len(str)+1)
	copy(buf, str)
	buf[len(str)] = 0

	// Return pointer to buffer
	return &buf[0]
}

// Helper function: Free C string
func FreeCString(cStr *uint8) {
	// In Go, we use slices to manage memory, so no need to free
	// This function is just to maintain API consistency
}
