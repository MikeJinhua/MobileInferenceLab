package com.mike.mobileinferencelab.temporalsr;

import android.content.Context;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Locale;

final class SpatialSrRunner {
    static final String MODEL_ASSET = "spatial_sr_xnnpack.pte";
    static final int INPUT_HEIGHT = 64;
    static final int INPUT_WIDTH = 64;
    private static final long[] INPUT_SHAPE = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    private static final long[] EXPECTED_OUTPUT_SHAPE = {1, 3, 128, 128};
    private static final double EXPECTED_CHECKSUM = 501.158030012808;
    private static final double CHECKSUM_TOLERANCE = 1.0e-3;

    private SpatialSrRunner() {}

    static Result run(Context context) throws IOException {
        File modelFile = copyAsset(context, MODEL_ASSET);
        float[] inputData = createDeterministicInput();
        Tensor inputTensor = Tensor.fromBlob(inputData, INPUT_SHAPE);

        try (Module module = Module.load(modelFile.getAbsolutePath())) {
            Tensor first = module.forward(EValue.from(inputTensor))[0].toTensor();
            Tensor second = module.forward(EValue.from(inputTensor))[0].toTensor();
            long[] outputShape = first.shape();
            if (!Arrays.equals(outputShape, EXPECTED_OUTPUT_SHAPE)) {
                throw new IllegalStateException(
                        "Unexpected output shape: " + Arrays.toString(outputShape));
            }
            float[] firstData = first.getDataAsFloatArray();
            float[] secondData = second.getDataAsFloatArray();
            if (firstData.length != 3 * 128 * 128 || firstData.length != secondData.length) {
                throw new IllegalStateException("Unexpected output element count: " + firstData.length);
            }

            float minimum = Float.POSITIVE_INFINITY;
            float maximum = Float.NEGATIVE_INFINITY;
            double checksum = 0.0;
            float repeatMaxDifference = 0.0f;
            for (int index = 0; index < firstData.length; index++) {
                float value = firstData[index];
                if (!Float.isFinite(value)) {
                    throw new IllegalStateException("Non-finite output at index " + index);
                }
                minimum = Math.min(minimum, value);
                maximum = Math.max(maximum, value);
                checksum += value;
                repeatMaxDifference = Math.max(
                        repeatMaxDifference, Math.abs(value - secondData[index]));
            }
            if (repeatMaxDifference > 1.0e-6f) {
                throw new IllegalStateException(
                        "Repeated inference differs by " + repeatMaxDifference);
            }
            double checksumDifference = Math.abs(checksum - EXPECTED_CHECKSUM);
            if (checksumDifference > CHECKSUM_TOLERANCE) {
                throw new IllegalStateException(
                        "Output checksum differs from eager reference by " + checksumDifference);
            }
            return new Result(
                    outputShape,
                    minimum,
                    maximum,
                    checksum,
                    repeatMaxDifference,
                    checksumDifference);
        }
    }

    private static float[] createDeterministicInput() {
        float[] values = new float[3 * INPUT_HEIGHT * INPUT_WIDTH];
        for (int index = 0; index < values.length; index++) {
            values[index] = ((index * 37) % 256) / 255.0f;
        }
        return values;
    }

    private static File copyAsset(Context context, String assetName) throws IOException {
        File output = new File(context.getFilesDir(), assetName);
        try (InputStream input = context.getAssets().open(assetName);
             FileOutputStream stream = new FileOutputStream(output, false)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) {
                stream.write(buffer, 0, count);
            }
        }
        return output;
    }

    static final class Result {
        final long[] outputShape;
        final float minimum;
        final float maximum;
        final double checksum;
        final float repeatMaxDifference;
        final double checksumReferenceDifference;

        Result(
                long[] outputShape,
                float minimum,
                float maximum,
                double checksum,
                float repeatMaxDifference,
                double checksumReferenceDifference) {
            this.outputShape = outputShape.clone();
            this.minimum = minimum;
            this.maximum = maximum;
            this.checksum = checksum;
            this.repeatMaxDifference = repeatMaxDifference;
            this.checksumReferenceDifference = checksumReferenceDifference;
        }

        String displayText() {
            return String.format(
                    Locale.US,
                    "ExecuTorch XNNPACK inference passed.\nInput: [1, 3, 64, 64]\nOutput: %s\nRepeat max difference: %.9g\nChecksum reference difference: %.9g\nRandom weights: pipeline validation only.",
                    Arrays.toString(outputShape),
                    repeatMaxDifference,
                    checksumReferenceDifference);
        }

        String logLine() {
            return String.format(
                    Locale.US,
                    "status=PASS output_shape=%s min=%.9g max=%.9g checksum=%.12g repeat_max_diff=%.9g checksum_ref_diff=%.9g",
                    Arrays.toString(outputShape),
                    minimum,
                    maximum,
                    checksum,
                    repeatMaxDifference,
                    checksumReferenceDifference);
        }
    }
}
