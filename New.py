import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
from PIL import Image,ImageTk

warnings.filterwarnings('ignore')

# Global dataframe to use in all tabs
df = None

# Function to select and load the CSV file
def select_file():
    global df
    file_path = filedialog.askopenfilename(title="Select a CSV File", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        df = pd.read_csv(file_path)
        messagebox.showinfo("File Loaded", "CSV file has been loaded successfully!")
    else:
        messagebox.showerror("Error", "No file selected!")

# Helper function to display plots within a tab
def display_plot(fig, tab):
    for widget in tab.winfo_children():
        widget.destroy()  # Clear previous widgets/plots
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Stage 1: Plot Bitcoin Closing Prices (also prepares 'year' column for Stage 4)
def run_stage_1():
    global df
    if df is not None:
        # Splitting the 'Date' column into 'year', 'month', 'day'
        if 'year' not in df.columns:
            splitted = df['Date'].str.split('-', expand=True)
            df['year'] = splitted[0].astype('int')
            df['month'] = splitted[1].astype('int')
            df['day'] = splitted[2].astype('int')

        # Convert the 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the Bitcoin Close Prices
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Close'])
        ax.set_title('Bitcoin Close Price', fontsize=15)
        ax.set_ylabel('Price in dollars')
        ax.set_xlabel('Date')
        display_plot(fig, trend_tab)
    else:
        messagebox.showerror("Error", "Please load a file first!")

# Stage 2: Distribution Plots
def run_stage_2():
    if df is not None:
        features = ['Open', 'High', 'Low', 'Close']
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, col in enumerate(features):
            sb.histplot(df[col], ax=axs[i], kde=True)
            axs[i].set_title(f'{col} Distribution')

        plt.tight_layout()
        display_plot(fig, distribution_tab)
    else:
        messagebox.showerror("Error", "Please load a file first!")

# Stage 3: Box Plots
def run_stage_3():
    if df is not None:
        features = ['Open', 'High', 'Low', 'Close']
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, col in enumerate(features):
            sb.boxplot(x=df[col], ax=axs[i])
            axs[i].set_title(f'{col} Boxplot')

        plt.tight_layout()
        display_plot(fig, price_variation_tab)
    else:
        messagebox.showerror("Error", "Please load a file first!")

# Stage 4: Bar Plot of Average Prices Grouped by Year
def run_stage_4():
    if df is not None:
        # Check that the 'year' column is present
        if 'year' not in df.columns:
            messagebox.showerror("Error", "'year' column is missing. Please run Stage 1 first!")
            return

        # Group by 'year' and calculate mean for each year
        data_grouped = df.groupby('year').mean()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()  # Flatten the array for easy indexing
        columns = ['Open', 'High', 'Low', 'Close']

        for i, col in enumerate(columns):
            data_grouped[col].plot(kind='bar', ax=axs[i])
            axs[i].set_title(f'Average {col} Price by Year')
            axs[i].set_ylabel('Price in dollars')  # Adding y-label for clarity

        plt.tight_layout()
        display_plot(fig, price_hike_tab)
    else:
        messagebox.showerror("Error", "Please load a file first!")

# Stage 5: Model Training and Evaluation
def run_stage_5():
    if df is not None:
        # Create additional features
        df['open-close'] = df['Open'] - df['Close']
        df['low-high'] = df['Low'] - df['High']
        df['is_quarter_end'] = np.where(df['month'].isin([3, 6, 9, 12]) & (df['day'] == 31), 1, 0)

        # Create the 'target' column
        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        # Check for the presence of the 'target' column
        if 'target' not in df.columns:
            messagebox.showerror("Error", "Target column 'target' is missing in the data!")
            return

        features = df[['open-close', 'low-high', 'is_quarter_end']]
        target = df['target']

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            features, target, test_size=0.1, random_state=2022)

        models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

        for model in models:
            model.fit(X_train, Y_train)
            print(f'{model} : ')
            print('Training Accuracy : ', metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
            print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
            print()

        # Confusion matrix for Logistic Regression
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, ax=ax)
        display_plot(fig, conclusion_tab)
    else:
        messagebox.showerror("Error", "Please load a file first!")

# Create main application window
root = tk.Tk()
root.title("Bitcoin Data Analysis Tool")
root.state('zoomed')  # Full-screen mode

# Styling
style = ttk.Style()
style.configure("TNotebook", tabposition='wn')  # Tabs on top
style.configure("TFrame", background="#f0f0f0")
style.configure("TButton", padding=5,background="pink")
style.configure("TLabel",font=("  Eurostile", 14))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TNotebook.Tab", padding=[10, 5], font=("Helvetica", 12, "bold"), background="#eaeadd", borderwidth=0)
style.map("TNotebook.Tab", background=[("selected", "#d5d5d5")])  # Change background color of selected tab

# Create a Tab Control
tab_control = ttk.Notebook(root)


# Tabs
home_tab = ttk.Frame(tab_control, padding="10 10 10 10")
trend_tab = ttk.Frame(tab_control, padding="10 10 10 10")
distribution_tab = ttk.Frame(tab_control, padding="10 10 10 10")
price_variation_tab = ttk.Frame(tab_control, padding="10 10 10 10")
price_hike_tab = ttk.Frame(tab_control, padding="10 10 10 10")
conclusion_tab = ttk.Frame(tab_control, padding="10 10 10 10")

#image for home tab

imageHome=Image.open("C:/Users/DELL/Desktop/my project/home.gif")
photo=ImageTk.PhotoImage(imageHome)
label_image3=ttk.Label(trend_tab,image=photo)
label_image3.photo=photo
label_image3.place(y=0,x=00)


# image for trend tab

imagetrend=Image.open("C:/Users/DELL/Desktop/my project/trend.png")
imagetrend = imagetrend.resize((1200, 690))
photo=ImageTk.PhotoImage(imagetrend)
label_image2=ttk.Label(trend_tab,image=photo)
label_image2.photo=photo
label_image2.place(y=0,x=00)


#image for distribution tab

image1=Image.open("C:/Users/DELL/Desktop/my project/distt.png")
image1 = image1.resize((1200, 690))
photo=ImageTk.PhotoImage(image1)
label_image1=ttk.Label(distribution_tab,image=photo)
label_image1.photo=photo
label_image1.place(y=0,x=00)

# image for price variation

image2=Image.open("C:/Users/DELL/Desktop/my project/pricea.png")
image2 = image2.resize((1200, 690))
photo=ImageTk.PhotoImage(image2)
label_image2=ttk.Label(price_variation_tab,image=photo)
label_image2.photo=photo
label_image2.place(y=0,x=00)

# price hike
image4=Image.open("C:/Users/DELL/Desktop/my project/price hike.png")
photo=ImageTk.PhotoImage(image4)
label_image4=ttk.Label(price_hike_tab,image=photo)
label_image4.photo=photo
label_image4.place(y=0,x=00)

# last tab
image5=Image.open("C:/Users/DELL/Desktop/my project/last.png")
photo=ImageTk.PhotoImage(image5)
label_image5=ttk.Label(conclusion_tab,image=photo)
label_image5.photo=photo
label_image5.place(y=0,x=00)


tab_control.add(home_tab, text='Home')
tab_control.add(trend_tab, text='Trend')
tab_control.add(distribution_tab, text='Distribution')
tab_control.add(price_variation_tab, text='Price Variation')
tab_control.add(price_hike_tab, text='Price Hike')
tab_control.add(conclusion_tab, text='Conclusion')
tab_control.pack(expand=1, fill='both')


file_entry = ttk.Entry(home_tab, width=50)
file_entry.place(y=600,x=50)
file_select_button = ttk.Button(home_tab, text="Select CSV File", command=select_file)
file_select_button.place(y=630,x=160)

# Trend Tab
trend_button = ttk.Button(trend_tab, text="Run Stage 1 (Close Price)", command=run_stage_1)
trend_button.pack(pady=20)

# Distribution Tab
distribution_button = ttk.Button(distribution_tab, text="Run Stage 2 (Distribution Plots)", command=run_stage_2)
distribution_button.place(y=600,x=160)

# Price Variation Tab
price_variation_button = ttk.Button(price_variation_tab, text="Run Stage 3 (Box Plots)", command=run_stage_3)
price_variation_button.place(y=600,x=160)

# Price Hike Tab
price_hike_button = ttk.Button(price_hike_tab, text="Run Stage 4 (Average Prices)", command=run_stage_4)
price_hike_button.place(y=630,x=160)

# Conclusion Tab
stage5_button = ttk.Button(conclusion_tab, text="Run Stage 5 (Model Training)", command=run_stage_5)
stage5_button.place(y=630,x=160)

# Run the application
root.mainloop()
