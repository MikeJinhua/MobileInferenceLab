package com.mike.mobileinferencelab.temporalsr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Locale;

final class ImageSrPipeline {
    static final int WIDTH = 64, HEIGHT = 64, SCALE = 2, WARMUP = 5, ITERATIONS = 20;
    private static final long[] INPUT_SHAPE = {1, 3, HEIGHT, WIDTH};
    private static final long[] OUTPUT_SHAPE = {1, 3, HEIGHT * SCALE, WIDTH * SCALE};
    private ImageSrPipeline() {}

    static Result run(Context context) throws IOException {
        long completeStart = now();
        Bitmap input = createSyntheticImage();
        long start = now();
        File modelFile = copyModel(context);
        double copyMs = ms(start);
        start = now();
        Module module = Module.load(modelFile.getAbsolutePath());
        double loadMs = ms(start);
        double[] pre = new double[ITERATIONS], infer = new double[ITERATIONS];
        double[] post = new double[ITERATIONS], e2e = new double[ITERATIONS];
        double[] bilinearTime = new double[ITERATIONS];
        Bitmap neural = null, bilinear = null;
        try {
            Tensor warmup = Tensor.fromBlob(bitmapToNchw(input), INPUT_SHAPE);
            for (int i = 0; i < WARMUP; i++) module.forward(EValue.from(warmup));
            for (int i = 0; i < ITERATIONS; i++) {
                long total = now();
                long stage = total;
                Tensor in = Tensor.fromBlob(bitmapToNchw(input), INPUT_SHAPE);
                pre[i] = ms(stage);
                stage = now();
                Tensor out = module.forward(EValue.from(in))[0].toTensor();
                infer[i] = ms(stage);
                if (!Arrays.equals(out.shape(), OUTPUT_SHAPE))
                    throw new IllegalStateException("Unexpected shape " + Arrays.toString(out.shape()));
                stage = now();
                neural = nchwToBitmap(out.getDataAsFloatArray(), 128, 128);
                post[i] = ms(stage);
                e2e[i] = ms(total);
            }
            for (int i = 0; i < ITERATIONS; i++) {
                start = now();
                bilinear = Bitmap.createScaledBitmap(input, 128, 128, true);
                bilinearTime[i] = ms(start);
            }
        } finally { module.close(); }
        return new Result(input, bilinear, neural, copyMs, loadMs, Stats.of(pre), Stats.of(infer),
                Stats.of(post), Stats.of(e2e), Stats.of(bilinearTime), ms(completeStart));
    }

    private static Bitmap createSyntheticImage() {
        int[] p = new int[WIDTH * HEIGHT];
        for (int y = 0; y < HEIGHT; y++) for (int x = 0; x < WIDTH; x++) {
            int r = x * 255 / 63, g = y * 255 / 63, b = (x ^ y) * 255 / 63;
            if (((x / 8) + (y / 8)) % 2 == 0) b = 255 - b;
            if (x == y || x + y == 63) r = g = b = 255;
            p[y * WIDTH + x] = Color.rgb(r, g, b);
        }
        return Bitmap.createBitmap(p, WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
    }

    private static float[] bitmapToNchw(Bitmap bitmap) {
        int plane = WIDTH * HEIGHT;
        int[] p = new int[plane];
        float[] out = new float[3 * plane];
        bitmap.getPixels(p, 0, WIDTH, 0, 0, WIDTH, HEIGHT);
        for (int i = 0; i < plane; i++) {
            out[i] = Color.red(p[i]) / 255f;
            out[plane + i] = Color.green(p[i]) / 255f;
            out[2 * plane + i] = Color.blue(p[i]) / 255f;
        }
        return out;
    }

    private static Bitmap nchwToBitmap(float[] v, int width, int height) {
        int plane = width * height;
        if (v.length != 3 * plane) throw new IllegalStateException("Unexpected element count");
        int[] p = new int[plane];
        for (int i = 0; i < plane; i++)
            p[i] = Color.rgb(toByte(v[i]), toByte(v[plane + i]), toByte(v[2 * plane + i]));
        return Bitmap.createBitmap(p, width, height, Bitmap.Config.ARGB_8888);
    }

    private static int toByte(float v) {
        if (!Float.isFinite(v)) throw new IllegalStateException("Non-finite output");
        return Math.round(Math.max(0f, Math.min(1f, v)) * 255f);
    }

    private static File copyModel(Context context) throws IOException {
        File file = new File(context.getFilesDir(), SpatialSrRunner.MODEL_ASSET);
        try (InputStream in = context.getAssets().open(SpatialSrRunner.MODEL_ASSET);
             FileOutputStream out = new FileOutputStream(file, false)) {
            byte[] buffer = new byte[8192]; int n;
            while ((n = in.read(buffer)) != -1) out.write(buffer, 0, n);
        }
        return file;
    }

    private static long now() { return SystemClock.elapsedRealtimeNanos(); }
    private static double ms(long start) { return (now() - start) / 1_000_000.0; }

    static final class Stats {
        final double mean, median, p95, minimum, maximum;
        Stats(double mean, double median, double p95, double minimum, double maximum) {
            this.mean = mean; this.median = median; this.p95 = p95;
            this.minimum = minimum; this.maximum = maximum;
        }
        static Stats of(double[] samples) {
            double[] s = samples.clone(); Arrays.sort(s); double sum = 0;
            for (double value : s) sum += value;
            double median = (s[9] + s[10]) / 2.0;
            return new Stats(sum / s.length, median, s[18], s[0], s[19]);
        }
        String text() {
            return String.format(Locale.US,
                    "mean %.3f, median %.3f, p95 %.3f, min %.3f, max %.3f ms",
                    mean, median, p95, minimum, maximum);
        }
    }

    static final class Result {
        final Bitmap input, bilinear, neural;
        final double copyMs, loadMs, completeMs;
        final Stats preprocess, inference, postprocess, neuralEndToEnd, bilinearTiming;
        Result(Bitmap input, Bitmap bilinear, Bitmap neural, double copyMs, double loadMs,
               Stats preprocess, Stats inference, Stats postprocess, Stats neuralEndToEnd,
               Stats bilinearTiming, double completeMs) {
            this.input = input; this.bilinear = bilinear; this.neural = neural;
            this.copyMs = copyMs; this.loadMs = loadMs; this.preprocess = preprocess;
            this.inference = inference; this.postprocess = postprocess;
            this.neuralEndToEnd = neuralEndToEnd; this.bilinearTiming = bilinearTiming;
            this.completeMs = completeMs;
        }
        String displayText() {
            return String.format(Locale.US,
                    "PASS — XNNPACK CPU float32, 64x64 → 128x128\nWarmup %d, samples %d\nModel copy %.3f; load %.3f ms\nPreprocess: %s\nInference: %s\nPostprocess: %s\nNeural E2E: %s\nBilinear: %s\nRandom weights: pipeline validation only.",
                    WARMUP, ITERATIONS, copyMs, loadMs, preprocess.text(), inference.text(),
                    postprocess.text(), neuralEndToEnd.text(), bilinearTiming.text());
        }
        String logLine() {
            StringBuilder result = new StringBuilder(String.format(Locale.US,
                    "status=PASS input=64x64 output=128x128 warmup=%d iterations=%d model_copy_ms=%.6f model_load_ms=%.6f",
                    WARMUP, ITERATIONS, copyMs, loadMs));
            appendStats(result, "preprocess", preprocess);
            appendStats(result, "inference", inference);
            appendStats(result, "postprocess", postprocess);
            appendStats(result, "neural_e2e", neuralEndToEnd);
            appendStats(result, "bilinear", bilinearTiming);
            return result.append(String.format(Locale.US, " complete_ms=%.6f", completeMs)).toString();
        }

        private static void appendStats(StringBuilder result, String name, Stats stats) {
            result.append(String.format(Locale.US,
                    " %s_mean_ms=%.6f %s_median_ms=%.6f %s_p95_ms=%.6f %s_min_ms=%.6f %s_max_ms=%.6f",
                    name, stats.mean, name, stats.median, name, stats.p95,
                    name, stats.minimum, name, stats.maximum));
        }
    }
}
