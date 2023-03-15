import pyautogui


screenWidth, screenHeight = pyautogui.size()
currentMouseX, currentMouseY = pyautogui.position()
print(currentMouseX, currentMouseY)
print(screenWidth, screenHeight)
# pyautogui.moveTo(int(screenWidth / 2), int(screenHeight / 2))
pyautogui.moveTo(500, 500)

print(pyautogui.position())
