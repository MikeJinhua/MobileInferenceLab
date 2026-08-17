package com.mike.mobileinferencelab.temporalsr;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final String TAG = "TNUImagePipeline";
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        int padding = Math.round(16.0f * getResources().getDisplayMetrics().density);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, padding, padding, padding);
        TextView status = addText(content, getString(R.string.phase3_ready), 16f);
        addText(content, "Original deterministic RGB — 64x64", 18f);
        ImageView inputView = addImage(content);
        addText(content, "Bilinear baseline — 128x128", 18f);
        ImageView bilinearView = addImage(content);
        addText(content, "Neural XNNPACK — 128x128", 18f);
        ImageView neuralView = addImage(content);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        setContentView(scroll);

        inferenceExecutor.execute(() -> {
            try {
                ImageSrPipeline.Result result = ImageSrPipeline.run(getApplicationContext());
                Log.i(TAG, result.logLine());
                runOnUiThread(() -> {
                    inputView.setImageBitmap(result.input);
                    bilinearView.setImageBitmap(result.bilinear);
                    neuralView.setImageBitmap(result.neural);
                    status.setText(result.displayText());
                });
            } catch (Throwable error) {
                Log.e(TAG, "status=FAIL", error);
                runOnUiThread(() -> status.setText("Pipeline failed: " + error.getMessage()));
            }
        });
    }

    private TextView addText(LinearLayout parent, String text, float size) {
        TextView view = new TextView(this);
        view.setText(text); view.setTextSize(size); view.setPadding(0, 8, 0, 8);
        parent.addView(view); return view;
    }

    private ImageView addImage(LinearLayout parent) {
        ImageView view = new ImageView(this);
        view.setAdjustViewBounds(true); view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        parent.addView(view, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                Math.round(220f * getResources().getDisplayMetrics().density)));
        return view;
    }

    @Override
    protected void onDestroy() {
        inferenceExecutor.shutdownNow();
        super.onDestroy();
    }
}
