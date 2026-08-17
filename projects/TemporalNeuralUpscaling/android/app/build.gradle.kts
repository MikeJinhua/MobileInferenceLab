plugins {
    id("com.android.application")
}

android {
    namespace = "com.mike.mobileinferencelab.temporalsr"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.mike.mobileinferencelab.temporalsr"
        minSdk = 23
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
}

dependencies {
    implementation("org.pytorch:executorch-android:1.3.1")
}
