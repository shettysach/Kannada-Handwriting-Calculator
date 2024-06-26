# #### Loading ConvNet model state

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.15)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.15)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32, momentum=0.15)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        logits = self.fc2(x)
        return logits


# In[3]:


model_path = "../model/ConvNet.pt"


# In[4]:


model = ConvNet()
model.load_state_dict(torch.load(model_path))

model.eval()


# #### Prediction function
# ##### Returns most probable class and its corresponding probability

# In[5]:


import cv2


# In[6]:


def predict(img):
    img = cv2.bitwise_not(img)
    img = cv2.resize(
        img, (28, 28), 
        interpolation = cv2.INTER_AREA
    ) 
    img = img.reshape(1, 1, 28, 28)
    img = img.astype('float32')
    img = img / 255.0
    img = torch.tensor(img)

    prediction = model(img)
    probabilities = F.softmax(prediction, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    prediction_probability = probabilities[0, predicted_class].item()

    return predicted_class, prediction_probability


# #### Function to change numbers to symbols
# ##### The model prediction returns 0-9 for numbers and 10-16 for symbols

# In[7]:


def number_to_symbol(x):
    if x == 10:
        return '+'
    if x == 11:
        return '-'
    if x == 12:
        return '*'
    if x == 13:
        return '/'
    if x == 14:
        return '('
    if x == 15:
        return ')'
    if x == 16:
        return '.'
    else:
        return str(x)


# #### Function to change English numbers to Kannada numbers

# In[8]:


kannada = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']

def english_to_kannada(s):
    return ''.join(
        kannada[int(i)] if i.isnumeric() 
        else i 
        for i in s
    )

result = english_to_kannada("( 3 * 4 ) / 6 - ")
print(result)


# #### Function to change displayed image

# In[9]:


def change_image(img):
    display_image = Image.fromarray(img)
    display_image = ctk.CTkImage(dark_image = display_image, size = (width // 2, height // 2))
    image_label.configure(image = display_image)
    image_label.image = display_image


# #### Function to solve expression
# ##### Since only numbers and symbols can be the input, eval() can be used safely

# In[10]:


def solve(predictions):
    expression = ''.join(
        predicted_class
        for predicted_class in predictions
    )
        
    exp.delete('1.0', ctk.END)
    sol.delete('1.0', ctk.END)

    try:
        solution = eval(str(expression)) 
        solution = str(float(f"{solution:.4f}"))
    
        print(f"{expression} = {solution}")
        exp.insert(ctk.INSERT, "{}    [ {}]".format(english_to_kannada(expression), expression))
        sol.insert(ctk.INSERT, "= {}    [ {} ]".format(english_to_kannada(solution), solution))
        
    except Exception:
        print(f"{expression} = Invalid")
        exp.insert(ctk.INSERT, "{}".format(expression))
        sol.insert(ctk.INSERT, "Invalid expression")


# #### Calculate function

# In[11]:


red = (0, 0, 225)
green = (0, 230, 0)
blue = (225, 0, 0)
white = (255, 255, 255)


# In[12]:


import numpy as np


# In[13]:


def calculate():
    img = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)

    # Add padding around the original image
    pad = 2
    img = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Blur it to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 5)

    # Inverted grayscale version with threshold (better for finding contours)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        grayscale, 200, 255, cv2.THRESH_BINARY
    )[1]

    # Finding contours (only external) and sort them by x co-ordinates, left to right
    threshold_img = cv2.bitwise_not(threshold_img)
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    predictions = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region of interest
        cropped_img = grayscale[y : y + h, x : x + w]

        # Adding padding if necessary for cases like "-",
        # where width is much larger than height
        if abs(w) > 1.5 * abs(h):
            pad = 3 * (w // h) ** 3
            cropped_img = cv2.copyMakeBorder(
                cropped_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255
            )

        # Resize and pad the cropped image
        resized_img = cv2.resize(cropped_img, (28, 28))
        padded_img = cv2.copyMakeBorder(
            resized_img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255
        )

        # Perform prediction
        predicted_class, prediction_probability = predict(padded_img)
        prediction_probability = round(prediction_probability * 100, 2)

        predictions.append(number_to_symbol(predicted_class))

        # Draw rectangle borders, predicted number/symbol and corresponding probability
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 7)

        # Adjusting text placement
        if y < 80:
            text_y = y + h + 85
        else:
            text_y = y - 25

        cv2.putText(
            img,
            f"{number_to_symbol(predicted_class)}",
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 0, 0),
            10,
        )
        cv2.putText(
            img,
            f"{prediction_probability}%",
            (x + 75, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.75,
            (0, 0, 255),
            3,
        )

    change_image(img)
    solve(predictions)


# #### Canvas
# ##### Setting width, height and of canvas, diameter and delta of the brush
# ##### Functions to paint and clear canvas
# ##### Depending on your screen resolution, you may need to change brush delta and thickness to get more accurate results

# In[14]:


width = 1700
height = 500


# In[15]:


def paint(event):
    delta = 10
    x1, y1 = (event.x - delta), (event.y - delta)
    x2, y2 = (event.x + delta), (event.y + delta)
    canv.create_oval(x1, y1, x2, y2, fill="black", width=17)
    draw.line([x1, y1, x2, y2], fill="black", width=17)


# In[16]:


def clear():
    canv.delete('all')
    draw.rectangle((0, 0, width, height), fill=(255, 255, 255, 0))
    exp.delete('1.0', ctk.END)
    sol.delete('1.0', ctk.END)


# #### GUI

# In[17]:


from PIL import ImageTk, Image, ImageDraw
import PIL
import customtkinter as ctk


# In[18]:


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

main_font = "Liberation Mono"
font_size = 30


# ##### To rerun the calculator, you need to just rerun the below cell. 

# In[19]:


root = ctk.CTk()
root.resizable(0, 0)
root.title("Handwriting Calculator")

canv = ctk.CTkCanvas(root, width=width, height=height, bg="white")
canv.grid(row=0, column=0, columnspan=2, padx=10, pady=17)
canv.bind("<B1-Motion>", paint)

image1 = Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

exp_font = ctk.CTkFont(family=main_font, size=font_size)
exp = ctk.CTkTextbox(
    root,
    exportselection=0,
    padx=10,
    pady=10,
    height=height // 4,
    width=width // 2,
    font=exp_font,
)
exp.grid(row=2, column=0, padx=0, pady=3)

sol_font = ctk.CTkFont(family=main_font, size=font_size, weight="bold")
sol = ctk.CTkTextbox(
    root,
    exportselection=0,
    padx=10,
    pady=10,
    height=height // 4,
    width=width // 2,
    font=sol_font,
    text_color="#3085ff",
)
sol.grid(row=3, column=0, padx=0, pady=3)

display_image = Image.new("RGB", (width, height), white)
display_image = ctk.CTkImage(dark_image=display_image, size=(width // 2, height // 2))

image_label = ctk.CTkLabel(root, image=display_image, text="")
image_label.grid(row=2, column=1, padx=10, pady=5, rowspan=2)

button_font = ctk.CTkFont(family=main_font, size=font_size-5)

Calculate = ctk.CTkButton(
    root,
    text="Calculate",
    command=calculate,
    fg_color="#0056C4",
    hover_color="#007dfe",
    font=button_font,
    height=height // 22.5,
)
 
Clear = ctk.CTkButton(
    root,
    text="Clear",
    command=clear,
    fg_color="#B50000",
    hover_color="#dd0000",
    font=button_font,
    height=height // 22.5,
)

Calculate.grid(row=1, column=0, padx=5, pady=1, sticky="ew")
Clear.grid(row=1, column=1, padx=5, pady=1, sticky="ew")

root.mainloop()

