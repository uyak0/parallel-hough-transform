# Install steps for OpenCV on Windows
https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_prebuilt
1. Download the pre-built OpenCV from the official site: https://opencv.org/releases/
2. Extract the downloaded file to a folder.
3. https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path
4. follow that link, do `setx OpenCV_DIR C:\THE\PATH\WHERE\YOU\EXTRACTED\OPENCV\build`
5. Open environment variable editor, add `%OPENCV_DIR%\bin` to the PATH variable.
6. https://docs.opencv.org/4.x/dd/d6e/tutorial_windows_visual_studio_opencv.html
7. follow the link for setting project properties in Visual Studio.
