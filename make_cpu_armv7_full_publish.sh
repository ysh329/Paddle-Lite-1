export NDK_ROOT=/opt/android-ndk-r17c
#./lite/tools/build.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_static full_publish
#./lite/tools/build.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_static test

#mkdir build.lite.android.armv7.gcc.full_publish
#cd build.lite.android.armv7.gcc.full_publish

./lite/tools/build.sh   --arm_os=android   --arm_abi=armv7   --arm_lang=gcc   --android_stl=c++_static   full_publish

#../lite/tools/ci_build.sh \
# --arm_os=android \
# --arm_abi=armv7 \
# --arm_lang=gcc cmake_arm


#make test_mobilenetv1_int8 -j1
