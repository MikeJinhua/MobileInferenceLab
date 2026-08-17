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

val spatialSrModel = layout.projectDirectory.file("src/main/assets/spatial_sr_xnnpack.pte")
val verifySpatialSrModel by tasks.registering {
    inputs.file(spatialSrModel)
    doLast {
        require(spatialSrModel.asFile.isFile) {
            "Missing generated model asset. Run: .venv-executorch\\Scripts\\python.exe -m tools.prepare_android_model"
        }
    }
}

tasks.named("preBuild") {
    dependsOn(verifySpatialSrModel)
}
