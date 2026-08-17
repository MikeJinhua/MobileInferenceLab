package com.mike.mobileinferencelab.temporalsr;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final String TAG = "TNUInference";
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView status = new TextView(this);
        status.setText(R.string.phase3_ready);
        status.setTextSize(20.0f);
        int padding = Math.round(24.0f * getResources().getDisplayMetrics().density);
        status.setPadding(padding, padding, padding, padding);
        setContentView(status);

        inferenceExecutor.execute(() -> {
            try {
                SpatialSrRunner.Result result = SpatialSrRunner.run(getApplicationContext());
                Log.i(TAG, result.logLine());
                runOnUiThread(() -> status.setText(result.displayText()));
            } catch (Throwable error) {
                Log.e(TAG, "status=FAIL", error);
                runOnUiThread(() -> status.setText("Inference failed: " + error.getMessage()));
            }
        });
    }

    @Override
    protected void onDestroy() {
        inferenceExecutor.shutdownNow();
        super.onDestroy();
    }
}
