# ![LOGO](https://github.com/M-Mabrouk/GhostWriter/blob/master/Required%20Multimedia/GP_Logo.png) GhostWriter 

> A simple way to write and and take notes freely without the need for a special electronic pen or tablet

![Demo GIF](https://github.com/M-Mabrouk/GhostWriter/blob/master/Required%20Multimedia/FCI_GIF.gif)

## Hardware Requirements:  
Intel RealSense D415/D435 Depth Camera

## Python Requirements: 
### modules:
* pyrealsense2  
* numpy  
* cv2  
* imutils  
* fpdf
* Google Cloud Vision

### version:  
* python 3.6

## How To Use
1. Plug the camera in your computer
2. Run main.py script
3. Specify the for corners of your paper in this order TopLeft -> TopRight -> BottomRight -> BottomLeft by pressing 'e' at each corner after the color is detected at this corner.
4. Start writing.
5. Press 'q' to quit, 'c' to clear, 's' to save as a pdf and ably ocr.

## Extra Notes  
1. No extra flags needed when executing the main.py script just plug the camera and run it
2. The script currently uses blue color detection to locate the pen until it is replaced with SDD object detection

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Dr. Ahmed Shawky Moussa - Project Supervisor
* Sally Samy - Designer
