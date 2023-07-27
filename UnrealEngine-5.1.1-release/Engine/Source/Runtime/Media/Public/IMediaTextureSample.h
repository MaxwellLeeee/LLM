// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"
#include "IMediaTimeSource.h"
#include "Math/Color.h"
#include "Math/IntPoint.h"
#include "Math/MathFwd.h"
#include "Math/Matrix.h"
#include "Math/Plane.h"
#include "Misc/Optional.h"
#include "Misc/Timecode.h"
#include "Misc/Timespan.h"
#include "Templates/SharedPointer.h"

#if WITH_ENGINE
	class FRHITexture;
	class IMediaTextureSampleConverter;
#endif


/**
 * Available formats for media texture samples.
 *
 * Depending on the decoder, the pixel data of a media texture frame may be stored
 * in one of the listed formats. Some of these may be supported natively by the
 * renderer, others may require a conversion step to a natively supported format.
 * The media texture sink is responsible for performing any necessary conversions.
 *
 * For details on the various YUV packings see: https://www.fourcc.org/yuv.php
 */
enum class EMediaTextureSampleFormat
{
	/** Format not defined. */
	Undefined,

	/** Four 8-bit unsigned integer components (AYUV packing) per texel. */
	CharAYUV,

	/** Four 8-bit unsigned integer components (Blue, Green, Red, Alpha) per texel. */
	CharBGRA,

	/** Four 10-bit unsigned integer components (Blue, Green, Red) & 2-bit alpha per texel. */
	CharBGR10A2,

	/**  Windows bitmap (like CharBGRA, but flipped vertically). */
	CharBMP,

	/** NV12 encoded monochrome texture with 8 bits per channel. */
	CharNV12,

	/** NV21 encoded monochrome texture with 8 bits per channel. */
	CharNV21,

	/** Four 8-bit unsigned integer components (UYVY packing aka. HDYC, IUYV, UYNV, Y422) per texel. */
	CharUYVY,

	/** Four 8-bit unsigned integer components (YUY2 packing aka. YUNV, YUYV) per texel. */
	CharYUY2,

	/** Four 8-bit unsigned integer components (YVYU packing) per texel. */
	CharYVYU,

	/** Three 16-bit floating point components (Red, Green, Blue) per texel. */
	FloatRGB,

	/** Four 16-bit floating point components (Red, Green, Blue, Alpha) per texel. */
	FloatRGBA,

	/** YUV v210 format which pack 6 pixel using 12 x 10bits components (128 bits block). */
	YUVv210,

	/** 4:4:4:4 AY'CbCr 16-bit little endian full range alpha, video range Y'CbCr. */
	Y416,

	/** DXT1. */
	DXT1,

	/** DXT5. */
	DXT5,

	/** YCoCg colour space encoded in DXT5. */
	YCoCg_DXT5,

	/** YCoCg colour space encoded in DXT5, with a separate alpha texture encoded in BC4. */
	YCoCg_DXT5_Alpha_BC4,
};

namespace MediaTextureSampleFormat
{
	 MEDIA_API const TCHAR* EnumToString(const EMediaTextureSampleFormat InSampleFormat);
};

/** Description of how the media texture sample is tiled (only used by tiled image sequences currently).*/
struct FMediaTextureTilingDescription
{
	FIntPoint TileNum = FIntPoint::ZeroValue;
	FIntPoint TileSize = FIntPoint::ZeroValue;
	int32 TileBorderSize = 0;

	FORCEINLINE bool IsValid() const
	{
		return TileNum.X > 0 && TileNum.Y > 0 && TileSize.X > 0 && TileSize.Y > 0;
	}
};

enum class EMediaOrientation
{
	Original = 0,
	CW90,
	CW180,
	CW270
};


/**
 * Interface for media texture samples.
 *
 * Media texture samples are generated by media players and sent to the registered
 * media texture sink. They contain a single frame of texture data along with extra
 * metadata, such as dimensions, time codes, and durations.
 *
 * Depending on the decoder, a frame's pixel data may be stored in a CPU memory
 * buffer, or in an RHI texture resource (only available when compiling against
 * the Engine). The media texture sample API supports both models via the GetBuffer
 * and the GetTexture methods. Sample instances need to implement only one of these.
 */
class IMediaTextureSample
{
public:
	/**
	 * Get the sample's frame buffer.
	 *
	 * The returned buffer is only valid for the life time of this sample.
	 *
	 * @return Buffer containing the texels, or nullptr if the sample holds an FTexture.
	 * @see GetDim, GetDuration, GetFormat, GetOutputDim, GetStride, GetTexture, GetTime
	 */
	virtual const void* GetBuffer() = 0;

	/**
	 * Get the width and height of the sample.
	 *
	 * The sample may be larger than the output dimensions, because
	 * of horizontal or vertical padding required by some formats.
	 *
	 * @return Buffer dimensions (in texels).
	 * @see GetBuffer, GetDuration, GetFormat, GetOutputDim, GetStride, GetTexture, GetTime
	 */
	virtual FIntPoint GetDim() const = 0;

	/**
	 * Get the number of mips encoded in the sample
	 *
	 * @return Number of mips in the sample (including base level)
	 * @note Default implementation provided as most samples will not feature mips
	 */
	virtual uint8 GetNumMips() const
	{
		return 1;
	}

	/**
	 * Get tile information (number, size and border size) of the sample.
	 *
	 * @return TileInfo struct
	 * @note Default implementation provided as most samples will not feature tiles
	 */
	virtual FMediaTextureTilingDescription GetTilingDescription() const
	{
		return FMediaTextureTilingDescription();
	}

	/**
	 * Get the amount of time for which the sample is valid.
	 *
	 * A duration of zero indicates that the sample is valid until the
	 * timecode of the next sample in the queue.
	 *
	 * @return Sample duration.
	 * @see GetBuffer, GetDim, GetFormat, GetOutputDim, GetStride, GetTexture, GetTime
	 */
	virtual FTimespan GetDuration() const = 0;

	/**
	 * Get the texture sample format.
	 *
	 * @return Sample format.
	 * @see GetBuffer, GetDim, GetDuration, GetOutputDim, GetStride, GetTexture, GetTime
	 */
	virtual EMediaTextureSampleFormat GetFormat() const = 0;

	/**
	 * Get the sample's desired output width and height.
	 *
	 * The output dimensions may be smaller than the frame buffer dimensions, because
	 * of horizontal and/or vertical padding that may be required for some formats.
	 *
	 * @return Output dimensions (in pixels).
	 * @see GetBuffer, GetDim, GetDuration, GetFormat, GetStride, GetTexture, GetTime
	 */
	virtual FIntPoint GetOutputDim() const = 0;

	/**
	 * Get the horizontal stride (aka. pitch) of the sample's frame buffer.
	 *
	 * @return The buffer stride (in number of bytes).
	 * @see GetBuffer, GetDim, GetDuration, GetFormat, GetOutputDim, GetTexture, GetTime
	 */
	virtual uint32 GetStride() const = 0;

#if WITH_ENGINE

	/**
	 * Get the sample's texture resource.
	 *
	 * @return Texture resource, or nullptr if the sample holds a frame buffer.
	 * @see GetBuffer, GetDim, GetDuration, GetFormat, GetOutputDim, GetStride, GetTime
	 */
	virtual FRHITexture* GetTexture() const = 0;

	/**
	 * Get media texture sample converter if sample implements it
	 *
	 * @return texture sample converter
	 */
	virtual IMediaTextureSampleConverter* GetMediaTextureSampleConverter()
	{ 
		return nullptr; 
	}

#endif //WITH_ENGINE

	/**
	 * Get the sample time (in the player's local clock).
	 *
	 * This value is used primarily for debugging purposes.
	 *
	 * @return Sample time.
	 * @see GetBuffer, GetDim, GetDuration, GetFormat, GetOutputDim, GetStride, GetTexture
	 */
	virtual FMediaTimeStamp GetTime() const = 0;

	/**
	 * Get the sample timecode if available.
	 *
	 * @return Sample timecode.
	 * @see GetTime
	 */
	virtual TOptional<FTimecode> GetTimecode() const { return TOptional<FTimecode>(); }

	/**
	 * Whether the sample can be held in a cache.
	 *
	 * Non-cacheable video samples become invalid when the next sample is available,
	 * and only the latest sample should be kept by video sample consumers.
	 *
	 * @return true if cacheable, false otherwise.
	 */
	virtual bool IsCacheable() const = 0;

	/**
	 * Whether the output of the sample is in sRGB color space.
	 *
	 * @return true if sRGB, false otherwise.
	 */
	virtual bool IsOutputSrgb() const = 0;

	/**
	 * Get image orientation vs. physically returned image data
	 *
	 * @return Image orientation
	 */
	virtual EMediaOrientation GetOrientation() const
	{
		return EMediaOrientation::Original;
	}

	/**
	 * Get pixel aspect ratio
	 *
	 * @return Pixel aspect ratio
	 */
	virtual double GetAspectRatio() const
	{
		FIntPoint OutputDim = GetOutputDim();
		return (double)OutputDim.X / (double)OutputDim.Y;
	}

	/**
	 * Get the ScaleRotation (2x2 matrix) for the sample.
	 *
	 * @return FLinearColor with xy = row 0 (dotted with U), zw = row 1 (dotted with V)
	 *
	 * @note For use with "external image" style output only. Use GetOrientation() otherwise
	 *
	 */
	virtual FLinearColor GetScaleRotation() const
	{
		return FLinearColor(1.0f, 0.0f, 0.0f, 1.0f);
	}

	/**
	 * Get the Offset applied after ScaleRotation for the sample.
	 *
	 * @return FLinearColor with xy = offset, zw must be zero
	 *
	 * @note For use with "external image" style output only
	 *
	 */
	virtual FLinearColor GetOffset() const
	{
		return FLinearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}

	/**
	 * Get the YUV to RGB conversion matrix.
	 *
	 * Equivalent to MediaShaders::YuvToRgbRec709Scaled Matrix. NOTE: previously in UE4 this was YuvToRgbRec601Scaled
	 *
	 * @return Conversion Matrix
	 */
	virtual const FMatrix& GetYUVToRGBMatrix() const
	{
		static const FMatrix DefaultMatrix(
			FPlane(1.16438356164f, 0.000000000000f, 1.792652263418f, 0.000000f),
			FPlane(1.16438356164f, -0.213237021569f, -0.533004040142f, 0.000000f),
			FPlane(1.16438356164f, 2.112419281991f, 0.000000000000f, 0.000000f),
			FPlane(0.000000f, 0.000000f, 0.000000f, 0.000000f)
		);

		return DefaultMatrix;
	}

	virtual void Reset() { }
	
public:

	/** Virtual destructor. */
	virtual ~IMediaTextureSample() { }
};