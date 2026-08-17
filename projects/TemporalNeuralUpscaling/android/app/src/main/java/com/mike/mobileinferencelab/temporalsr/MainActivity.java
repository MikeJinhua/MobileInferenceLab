package com.mike.mobileinferencelab.temporalsr;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public final class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView status = new TextView(this);
        status.setText(R.string.phase3_ready);
        status.setTextSize(20.0f);
        int padding = Math.round(24.0f * getResources().getDisplayMetrics().density);
        status.setPadding(padding, padding, padding, padding);
        setContentView(status);
    }
}
