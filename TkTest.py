# from tkinter import *
import tkinter as tk
from tkinter import ttk

def hey():
    print("HEY!")

window = tk.Tk()
window.geometry('300x200')

button = ttk.Button(window, text="Hey", command=hey)
button.pack()
# class TkTest:
#     def __init__(self):
#         self.root = Tk()
#         self.geometry("200x200")
#
#         btn = Button(self,
#                      text="Click to open a new window",
#                      command=self.hey)
#
#         # def openNewWindow():
#         #     newWindow = TopLevel(master)
#
window.mainloop()

#
#
# #
# import tkinter as tk
# from tkinter import ttk
#
# # root window
# root = tk.Tk()
# root.geometry('300x200')
# root.resizable(False, False)
# root.title('Button Demo')
#
# # exit button
# exit_button = ttk.Button(
#     root,
#     text='Exit',
#     command=lambda: root.quit()
# )
#
# exit_button.pack(
#     ipadx=5,
#     ipady=5,
#     expand=True
# )
#
# root.mainloop()