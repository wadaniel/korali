import tkinter as tk
from tkinter import *
from tkinter import ttk # if python.7 Tkinter. ttk is like the CSS.
from tkinter.messagebox import showerror, showwarning, showinfo
import os, sys
import json
import copy                     # recursiveUpdate
from functools import partial # Callback to retrieve values from entries



# ************ VARIABLES *****************
HUGE_FONT = ("Verdana",14)
LARGE_FONT = ("Verdana",12) #Font and size.
NORM_FONT = ("Verdana",10) #Font and size.
SMALL_FONT = ("Verdana",8) #Font and size.
RES_FONT = ('Courier',12)

darkColor ='lightseagreen'
lightColor = '#00A3E0'
extraColor = '#183A54'

datcounter = 0

contador = 1
numofexperiments = [] # Empty list to keep the count of tabs.
contadorVariables = 1

font = "Helvetica 10"

menus = []

totalTabs = ''
experiments = {}
selectedtab = ''

default = False
on = 0

mainPath = '../../source/modules/problem'
mainPath2 = '../../source/modules/solver'

printVariable = 'Configuration Settings'

variables = {}

#############################################################
## ******************* FUNCTIONS ****************************


# ******************* TUTORIAL ******************
def tutorial():
    
    def page2():
        tut.destroy()
        tut2 = tk.Tk()

        def page3():
            tut2.destroy() # Destroys page 2 and opens page 3:
            tut3 = tk.Tk()

            tut3.wm_title('Part 3!')

            label = ttk.Label(tut3, text = 'Part 3', font = NORM_FONT)
            label.pack(side='top', fill='x', pady=10)
            B1 = ttk.Button(tut3, text='Done!', command=tut3.destroy)
            B1.pack()
            tut3.mainloop()

        tut2.wm_title('Part 2!')
        label = ttk.Label(tut2, text = 'Part 2', font = NORM_FONT)
        label.pack(side='top', fill='x', pady=10)
        B1 = ttk.Button(tut2, text='Done!', command= page3)
        B1.pack()
        tut2.mainloop()

    tut = tk.Tk()
    tut.wm_title('Tutorial')
    label = ttk.Label(tut, text='What do you need help with?', font = NORM_FONT)
    label.pack(side='top', fill='x', pady=10)

    B1 = ttk.Button(tut, text = 'Overview of the application', command = page2)
    B1.pack()
    B2 = ttk.Button(tut, text = 'How do I get Days-timed data?', command = lambda:popupmsgwarning('Not yet complited')) # How do I trade???
    B2.pack()
    B3 = ttk.Button(tut, text = 'Graph Question/Help', command = lambda: popupmsgwarning('Not yet complited'))
    B3.pack()

    tut.mainloop()
## -------------------------END OF TUTORIAL FUNCTION ----------------
     

############################### QUICK FUNCTIONS:
def popupmsgwarning(text):
    Tk().withdraw()
    showwarning(title = 'Error',message=text)
    
def openFile(fileName):
	print("Open File "+fileName+"!")

def validardigit(num):
    # Función para validar POSITIVOS Y NEGATIVOS.
    r = False
    if num == '' or num == '-':
        r = True
    elif num[0] == '-':
        if num[1::].isdigit():
            r= True
    else:
        if num.isdigit():
            r = True
        else:
            r = False
    return r


def validarstring(string_Var):
    r = False
    if string_Var.isalpha() or string_Var == '' or ' ' in string_Var:
        r = True
    return r

def on_tab_selected(event):
    global selectedtab
    tabb = event.widget.select()
    selectedtab = event.widget.tab(tabb,'text')
    
def getNextVariableId(groupVariables):
    pass

def getNextTabId():
    global totalTabs

    for item in totalTabs.widget:
        pass

def explainDescription(description):
    tut = tk.Tk()
    half = len(description)/2
    tut.wm_title('Description')
    label = ttk.Label(tut, text= description[0:int(half)]+'\n'+description[int(half)::], font = NORM_FONT)
    label.pack(side='top', fill='x', pady=10)
    tut.maxsize('900','300')

def getValues(entry, f):
    global selectedtab # To know at which experiment we are.
    global experiments
    name = {}
    entries = experiments[selectedtab]['entry']
    for entry_label in  entries.keys():
        entry = entries[entry_label]
        print(entry_label +'=' + entry.get())
        

def callback(var, *args):
    print(var.get())
    
################
################ DELETE FUNCTIONS:


def deleteTab(self,totalTabs):
    global contador
    if contador >2:
        for item in totalTabs.winfo_children():
            if str(item)==totalTabs.select():
                item.destroy()
                return
    else:
        popupmsgwarning('At least 2 experiments to delete 1')

def deleteVariable(self):
    global contadorVariables
    global experiments
    global selectedtab
    
    totalTabs2 = experiments[selectedtab]['totalTabs2']
    ## Delete Tab and frame inside:
    if len(totalTabs2.tabs()) > 1:
        totalTabs2.forget(totalTabs2.select())
    else:
        popupmsgwarning('At least 2 variables to delete 1')

#################
################# PRINTING FUNCTIONS:

def printVariables(self,directorio,DB,cont):
    '''
    linktoVariables refers to the .config path of each element of the cascade. This path also works
    as a key in 'variables' dictionary, and as a value, we can find their variables configuration.
    '''
    global on
    global experiments
    global variables
    global linktoVariables0
    global linktoVariables1
    
    
    # IF AND ONLY IF WE HAVE A PROBLEM AND A SOLVER SET...
    if (cont == 0 and on == 1) or (cont == 0 and on == 0):
        linktoVariables0 = DB[directorio]['config']
        on += -1
    elif cont == 0 and on == -1:
        linktoVariables0 = DB[directorio]['config']
    elif (cont == 1 and on == -1) or (cont == 1 and on == 0):
        linktoVariables1 = DB[directorio]['config']
        on += 1
    elif cont == 1 and on == 1:
        linktoVariables1 = DB[directorio]['config']
        
    if on == 0:
        linktoVariables = []
        linktoVariables.append(linktoVariables0)    # Añadimos las variables que vienen del Problem.
        linktoVariables.append(linktoVariables1)    # Añadimos las variables que vienen del Solver.
        #cont = 2
        #printConfig(directorio,DB, cont)
    
        ro = 0
        co = 0

        double_Var = tk.DoubleVar()
        string_Var = tk.StringVar()
        num = tk.IntVar()
        boolean = tk.BooleanVar()
        self.entryVar = {}
        
        for part in linktoVariables:
            for llave in variables[part].keys():
                #print(llave)
                if llave == 'Type':
                # llave son las llaves del diccionario, por ejemplo: Type, Function, Description, Produced By...
                    texto = variables[part]['Name']
                    texto = texto[0] # Remove the '{}' from the label name.
                    description = variables[part]['Description']
                    if len(description)>45:
                        fakedescription = str(description[0:42])+'...'
                    else:
                        fakedescription = description
                    #print(variables[part])
                    if 'vector' not in variables[part][llave]:
                                        
                        if 'string' in variables[part][llave] or 'korali' in variables[part][llave]: # == 'std::string
                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='floralwhite', anchor="w", font="Arial 16")
                            self.label.grid(row=ro, column=co,pady = 20,padx=20, sticky='w')
                            self.entryVar[texto] = tk.Entry(self,width=10,textvariable = string_Var)
                            self.entryVar[texto].grid(row=ro, column=co+1, pady= 20, padx=20, sticky = 'w')
                            self.corrector = self.register(validarstring)
                            self.entryVar[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
                            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = 'azure',activebackground = 'white',
                                                command = lambda:explainDescription(description))
                            self.buttonLabel.grid(row=ro, column=co+3,sticky='w')
                            #self.label2=tk.Label(self, text = descripcion,bg='azure',relief = 'raised', font = font, width = 60)
                            #self.label2.grid(row=r, column=c+3,sticky='w')
                            #self.entryVar[texto].trace('w', partial(callback, key,self.entryVar[texto] ))
                                            
                        elif variables[part][llave] == 'double':
                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='floralwhite', anchor="w", font="Arial 16")
                            self.label.grid(row=ro, column=co,pady = 20,padx=20, sticky='w')
                            self.spinbox = Spinbox(self, width=10, from_=0, to=9999, wrap=True, textvariable=num, state='normal')
                            self.spinbox.grid(row=ro, column=co+1, pady= 20, padx=20)
                            self.corrector = self.register(validardigit)
                            self.spinbox.config(validate = 'key',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
                            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = 'azure',activebackground = 'white',
                                                command = lambda:explainDescription(description))
                            self.buttonLabel.grid(row=ro, column=co+3,sticky='w')#spinbox_value = spinbox.get()
                                        
                        elif variables[part][llave] == 'bool':
                            self.label=tk.Label(self, text=texto, width = 17,justify='left',bg='floralwhite', anchor="w", font="Arial 16")
                            self.label.grid(row=ro, column=co,pady = 20,padx=20, sticky='w')
                            self.true = tk.Radiobutton(self, width=10,text='Yes',highlightbackground = 'azure',activeforeground ='white', bg='floralwhite',
                                                        activebackground='darkcyan',borderwidth = 0,variable=boolean, value='t')
                                            
                            self.false = tk.Radiobutton(self,width=10, text='No',activeforeground = 'white',bg='floralwhite',
                                                        activebackground='lightcoral',highlightbackground = 'azure',borderwidth = 0,variable=boolean, value='f')
                                            
                            self.true.grid(row=ro, column=co+1, columnspan = 1,pady= 20, padx=20, sticky = 'w')
                            self.false.grid(row=ro, column=co+2, pady= 20, columnspan= 1, padx=20, sticky = 'w')
                                            
                    else:
                        if variables[part][llave] == 'double':
                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='floralwhite', anchor="w", font="Arial 16")
                            self.label.grid(row=ro, column=co,pady = 20,padx=20, sticky='w')
                            self.entry_vector_double = Entry(self, width=10,textvariable = string_Var)
                            self.entry_vector_double.grid(row=ro, column=1, pady= 20, padx=20)
                            self.corrector = self.register(validardigit)
                                            
                    ro +=1
        
    else:
        for widget in self.winfo_children():
            widget.destroy()
    #print(on)
    
    
    
#printConfig(directorio,DB, cont)
def printConfig(directorio,DB, cont):
    
    global printVariable
    global selectedtab

    # From which cascade do we come from? Problem, Solver or Variable?
    if cont == 0:
        self = experiments[selectedtab]['rightFrame']
    elif cont == 1:
        self = experiments[selectedtab]['bottomleftFrame']
    elif cont == 2:
        self = experiments[selectedtab]['bottomrightFrame']
    #####

    ## DELETE ANY WIDGET THAT WAS THERE BEFORE:
    for widget in self.winfo_children():
        widget.destroy()
    
    # Set variables:
    r = 0
    c = 0

    # Start reading and printing on screen:
    self.titulo=tk.Label(self, text=list(DB.keys())[0] ,justify='left', anchor="w", font="Arial 18", bg = 'darkcyan', fg='white')
    self.titulo.grid(row=r, column=c,pady = 10 ,padx=20, sticky='w', columnspan = 4)
    config = DB[directorio]['herencia']
    conf_sett = 'Configuration Settings'
    self.entry = {}
    experiments[selectedtab]['entry'] = self.entry
    self.stringVar = {}
    double_Var = tk.DoubleVar()
    num = tk.IntVar()
    boolean = tk.BooleanVar()
    
    if conf_sett in config.keys():
    # Loop through herencia.
        for key in config.keys():
            if key == conf_sett:
                lista = config[key]
                for dicc in lista:
                    if type(dicc) == dict:
                        if 'Type' not in dicc.keys():
                            popupmsgwarning("No 'Type' field found")
                            break
                        else:  
                            for key2 in dicc.keys():
                                if key2 == 'Type':
                                    # key2 son las llaves del diccionario, por ejemplo: Type, Function, Description, Produced By...
                                    texto = dicc['Name']
                                    texto = texto[0] # Remove the {} from the label name.
                                    description = dicc['Description']
                                    if len(description)>45:
                                        fakedescription = str(description[0:42])+'...'
                                    else:
                                        fakedescription = description
                                    
                                    if 'vector' not in dicc[key2]:
                                        #print(dicc[key2])
                                        
                                        if 'string' in dicc[key2] or 'korali' in dicc[key2]: # == 'std::string
                                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='azure', anchor="w", font="Arial 16")
                                            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
                                            self.stringVar[texto] = tk.StringVar()
                                            #self.stringVar[texto].trace('w', partial(callback,self.stringVar[texto]))
                                            self.entry[texto] = tk.Entry(self,width=10,textvariable = self.stringVar[texto])
                                            self.entry[texto].grid(row=r, column=c+1, pady= 20, padx=20, sticky = 'w')
                                            self.corrector = self.register(validarstring)
                                            self.entry[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
                                            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = 'azure',activebackground = 'white',
                                                                         command = lambda:explainDescription(description))
                                            self.buttonLabel.grid(row=r, column=c+3,sticky='w')
                                            #self.label2=tk.Label(self, text = descripcion,bg='azure',relief = 'raised', font = font, width = 60)
                                            #self.label2.grid(row=r, column=c+3,sticky='w')
                                            
                                        elif dicc[key2] == 'double':
                                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='azure', anchor="w", font="Arial 16")
                                            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
                                            self.spinbox = Spinbox(self, width=10, from_=0, to=9999, wrap=True, textvariable=num, state='normal')
                                            self.spinbox.grid(row=r, column=c+1, pady= 20, padx=20)
                                            self.corrector = self.register(validardigit)
                                            self.spinbox.config(validate = 'key',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
                                            #spinbox_value = spinbox.get()
                                        
                                        elif dicc[key2] == 'bool':
                                            self.label=tk.Label(self, text=texto, width = 17,justify='left',bg='azure', anchor="w", font="Arial 16")
                                            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
                                            self.true = tk.Radiobutton(self, width=10,text='Yes',highlightbackground = 'azure',activeforeground ='white', bg='azure',
                                                                       activebackground='darkcyan',borderwidth = 0,variable=boolean, value='t')
                                            
                                            self.false = tk.Radiobutton(self,width=10, text='No',activeforeground = 'white',bg='azure',
                                                                        activebackground='lightcoral',highlightbackground = 'azure',borderwidth = 0,variable=boolean, value='f')
                                            
                                            self.true.grid(row=r, column=c+1, columnspan = 1,pady= 20, padx=20, sticky = 'w')
                                            self.false.grid(row=r, column=c+2, pady= 20, columnspan= 1, padx=20, sticky = 'w')
                                            
                                        elif dicc[key2] == 'size_t':
                                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='azure', anchor="w", font="Arial 16")
                                            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
                                            self.stringVar[texto] = tk.StringVar()
                                            self.entry[texto] = tk.Entry(self,width=10,textvariable = self.stringVar[texto])
                                            self.entry[texto].grid(row=r, column=c+1, pady= 20, padx=20, sticky = 'w')
                                            self.corrector = self.register(validarstring)
                                            self.entry[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
                                            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = 'azure',activebackground = 'white',
                                                                         command = lambda:explainDescription(description))
                                            self.buttonLabel.grid(row=r, column=c+3,sticky='w')
                                            
                                            
                                    else:
                                        if dicc[key2] == 'double':
                                            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg='azure', anchor="w", font="Arial 16")
                                            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
                                            self.stringVar[texto] = tk.StringVar()
                                            self.entry[texto] = tk.Entry(self,width=10,textvariable = self.stringVar[texto])
                                            #self.entry_vector_double = Entry(self, width=10,textvariable = string_Var)
                                            self.entry[texto].grid(row=r, column=1, pady= 20, padx=20)
                                            self.corrector = self.register(validardigit)
                                            
                                r+=1
                                
        #https://stackoverflow.com/questions/4140437/interactively-validating-entry-widget-content-in-tkinter
                    elif type(dicc) == list:
                        print(dicc)
    else:
        popupmsgwarning('No Configuration Settings found')
    
        
    self.button_done = tk.Button(self, text = 'Done!',bd = 3,command = lambda: getValues(self.entry,self))
    self.button_done.grid(row=0, column=1, pady= 20, padx=20, sticky = 'W')
    self.update_idletasks
                        
##########################
########################## MAIN FUNCTIONS:
    
def recursiveUpdate(dest, defaults):
 if (isinstance(defaults, dict)):               # Si default es un diccionario...
  for k, x in defaults.items():                 # Para cada key y valor.
   if (not k in dest): dest[k] = copy.deepcopy(defaults[k])
   else: 
     recursiveUpdate(dest[k], defaults[k])
 if (isinstance(defaults, list)):
  for k in range(len(defaults)):
   if (not k in dest): dest.append(copy.deepcopy(defaults[k]))
   else: 
     recursiveUpdate(dest[k], defaults[k])


     
def splitPath(s):
    dirs = []
    dirName = ''
    for i in s:
        if i == os.path.sep:
            dirs += [dirName]
            dirName = ''
        else:
            dirName += i
    if len(dirName) > 0:
        dirs += [dirName]
    return dirs

def readDirs(filePath,configTreeDB,default):
    dirs = splitPath(filePath)
    levels=len(dirs)
    sublevel = levels + 1
    dirInfoDic={}
    childrenList=[]
    dest = False
    for (dirPath,dirNames,fileNames) in os.walk(filePath):
        if dirPath == filePath:
            for fileName in fileNames:
                if fileName.endswith('.config'):
                    dirName = dirs[levels-1]
                    configTreeDB[dirName] = dirInfoDic
                    dirInfoDic['config'] = filePath + os.path.sep + fileName
                    dirInfoDic['children'] = childrenList
                    try:
                        archivo = open(filePath+'/'+fileName,'r')
                        file = archivo.read()
                        dest = json.loads(file)
                        if default:
                            recursiveUpdate(dest,default)
                        dirInfoDic['herencia'] = dest
                        #print(dirInfoDic['config'])
                        #print(dirInfoDic['herencia'],'\n')
                    finally:
                        archivo.close()                                        
                elif fileName.endswith('.rst'): # Dame el nombre dentro del README:
                    try:
                        file = open(filePath+'/'+fileName,"r")
                        lines = file.readlines()
                        for i in range(0,len(lines)):
                            line = lines[i]
                            if '******' not in line:
                                continue
                            else:
                                name = lines[i+1]
                                dirInfoDic['names'] = name.replace('\n','') # Quitar '\n'
                                break
                    finally:
                        file.close() ## Cierra el archivo README.
            continue
        dirs=splitPath(dirPath)
        levels=len(dirs)
        if levels > sublevel:
            del dirNames[:]
            continue
        readDirs(dirPath, configTreeDB, dest)
        dirName = dirs[levels - 1]
        childrenList += [dirName]
    crearVariables(configTreeDB)
    
### CREAR EL DICCIONARIO DE VARIABLES COMBINANDO LOS DOS DICCIONARIOS ANTERIORES
def crearVariables(diccionario):
    for key in diccionario.keys():
        if 'Variables Configuration' in diccionario[key]['herencia'].keys():
            if diccionario[key]['herencia']['Variables Configuration']:
                for key2 in diccionario[key]['herencia']['Variables Configuration']:
                    variables[diccionario[key]['config']] = key2
                    
def crearMenu(padre,directorio,DB,cont):
    global menus
    global experiments
    
    bottomrightFrame = experiments[selectedtab]['bottomrightFrame']
    nombre = DB[directorio]['names']
    if directorio not in menus: # Checkeamos si el directorio está en la lista de menus.
        subMenu = Menu(padre,activeforeground = 'teal')
        padre.add_cascade(label=nombre, menu = subMenu)
        dirInfo=DB[directorio]
        configPath=dirInfo['config']
        if configPath != "NULL" and len(dirInfo['children']) == 0: # SI ES UN LEAF:
            dirs=splitPath(configPath)
            subMenu.add_command(label = dirs[len(dirs)-1], command = lambda : [printConfig(directorio,DB, cont),printVariables(bottomrightFrame.tab2,directorio,DB,cont)])
        children=dirInfo['children']
        for child in children:
            crearMenu(subMenu,child,DB,cont)
    menus.append(directorio)



def cascade(mainPath,DB,cont):
    global experiments
    dirs = splitPath(mainPath)
    for directorio in DB.keys():
        if directorio in dirs:
            titulo = directorio
            break
    leftFrame = experiments[selectedtab]['leftFrame']
                
    menuButton = tk.Menubutton(leftFrame, text = DB[titulo]['names'], indicatoron=True, borderwidth = 1, relief='raised', width=20, border=3)
    menuPadre = tk.Menu(menuButton, tearoff=False,activeforeground = 'teal')
    menuButton.configure(menu=menuPadre)
    menuButton.grid(row=0, column = cont, pady=10, padx=10, rowspan = 3)
        
    menus.append(titulo) # Añadimos el título del boton, para no repetirlo en los desplegables.
      
    # Empezar los menus recursivamente llamando a la funcion crearMenu:
    for directorio in DB.keys():
        crearMenu(menuPadre,directorio,DB,cont)
            
    menus.clear() # Allows creating different experiments by emptying the menus list.




def crearFrameVariables():
    global contadorVariables
    global experiments
    global selectedtab
    bottomrightFrame = experiments[selectedtab]['bottomrightFrame']
    totalTabs2 = experiments[selectedtab]['totalTabs2']
    #for item in self.totalTabs
    if contadorVariables < 15:
        ## Create Tab and frames inside:
        bottomrightFrame.tab2 = tk.Frame(bottomrightFrame, height = 450, width = 300, background = 'floralwhite')#florawhite
        totalTabs2.add(bottomrightFrame.tab2, text = 'Variable '+str(contadorVariables))
        contadorVariables += 1
        bottomrightFrame.tab2.grid_propagate(0)
    else:
        popupmsgwarning('Number of Variables exceeded!')
    
## Crea los tabs de uno en uno
def crearTab(self,totalTabs):
    global contador
    global experiments
    global selectedtab

    if contador < 7:
                ## Create Tab and frames inside:
                self.tab = tk.Frame(self.totalTabs)
                self.buttonTab = tk.Button(self.tab,text = 'xxxx')
                self.buttonTab.grid(row=0,column=0, sticky='e')
                selectedtab = 'Experiment '+str(contador)
                self.totalTabs.add(self.tab, text = selectedtab)
                self.totalTabs.bind('<<NotebookTabChanged>>', on_tab_selected)

                self.leftFrame = tk.Frame(self.tab, width = 950, height = 80, background ='azure', borderwidth = 3, relief = 'solid')
                self.leftFrame.grid(column = 0, row = 0, sticky = 'nsew',rowspan = 1)


                self.rightFrame = tk.Frame(self.tab, width = 900, height = 450,background ='azure', borderwidth = 1,relief = 'groove')
                self.rightFrame.grid(column = 1, row = 0, stick = 'nsew',rowspan = 2) #, sticky = 'ne', rowspan = 1, columnspan = 1)


                self.bottomleftFrame1 = tk.Frame(self.tab, background = 'azure', width = 950, height = 850, borderwidth = 3,relief = 'groove')#width = 950, height = 850
                self.bottomleftFrame1.grid(column = 0, row = 1, sticky = 'nsew',rowspan = 2) #, sticky = 'sw', rowspan = 3, columnspan = 1)
                self.bottomleftFrame1.grid_propagate(0)
                ## CANVAS + SCROLLBAR

                ## Good One:
                '''
                yscrollbar = tk.Scrollbar(self.bottomleftFrame1, bg = 'darkcyan',orient = 'vertical', width = 10)
                self.canvas = tk.Canvas(self.bottomleftFrame1, bg = 'azure',
                yscrollcommand=yscrollbar.set, width = 930, height = 840)
                self.canvas.grid(sticky='nsew')
                yscrollbar.config(command = self.canvas.yview)
                yscrollbar.grid(row=0, column=10, sticky=N+S)

                self.canvas.configure(scrollregion=(0,0,2000,2000))
                #self.bottomleftFrame1.config(scrollregion=self.canvas.bbox("all"))
                '''
                self.canvas=Canvas(self.bottomleftFrame1,width = 950, height = 850,bg = 'azure', scrollregion= (0,0,1500,1500)) #yscrollcommand = vbar.set
                #self.canvas.pack(side=LEFT,expand=True,fill=BOTH)
                hbar=Scrollbar(self.bottomleftFrame1,orient=HORIZONTAL)
                hbar.pack(side=BOTTOM,fill=X)
                hbar.config(command=self.canvas.xview)
                vbar=Scrollbar(self.bottomleftFrame1,orient=VERTICAL)
                vbar.pack(side=RIGHT,fill=Y)
                vbar.config(command=self.canvas.yview)
                self.canvas.pack(side=LEFT,expand=True,fill=BOTH)
                self.canvas.config(width=950,height=850)
                self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
                
                
                self.canvas.grid_propagate(0)

                '''
                vbar=Scrollbar(self.bottomleftFrame1,orient=VERTICAL,bg = 'darkcyan', width = 10)
                vbar.pack(side=RIGHT,fill=Y)
                self.canvas=Canvas(self.bottomleftFrame1, yscrollcommand = vbar.set,width = 950, height = 850, bg = 'azure', scrollregion= (0,0,2000,2000))
                self.canvas.pack(side=LEFT,expand=True,fill=BOTH)
                #self.canvas.config(width=300,height=300)
                vbar.config(command=self.canvas.yview)
                self.canvas.config(yscrollcommand=vbar.set, scrollregion = self.canvas.bbox("all"))
                '''
                ## END CANVAS + SCROLLBAR
                

                self.bottomrightFrame = tk.Frame(self.tab, width = 900, height = 450, background ='azure', borderwidth = 1,relief = 'ridge')
                self.bottomrightFrame.grid(column = 1, row = 2,sticky = 'nsew', rowspan = 1)#, sticky = 'se', rowspan = 2, columnspan = 1)

                
                # STORE A DICTIONARY WITH ALL THE FRAMES ON IT TO BE CALLED ON ANY FUNCTION AND AVOID PASSING FRAMES.
                expElements = {}

                experiments[selectedtab] = expElements
                expElements['leftFrame'] = self.leftFrame
                expElements['rightFrame'] = self.rightFrame
                expElements['bottomrightFrame'] = self.bottomrightFrame
                expElements['bottomleftFrame'] = self.canvas

                self.totalTabs2 = ttk.Notebook(self.bottomrightFrame)
                self.totalTabs2.pack(expand = 1, fill = "both")
               
                expElements['totalTabs2'] = self.totalTabs2

                crearFrameVariables()
                

                self.leftFrame.grid_propagate(0)
                self.rightFrame.grid_propagate(0)
                #self.bottomleftFrame.grid_propagate(0)
                self.bottomrightFrame.grid_propagate(0)
                #self.bottomleftFrame1.grid_propagate(0)
                

               # LEFT SIDE OF TAB1:
                
                ## CASCADE:
                # Elegir el título de los Menús:
                cont= 0
                cascade(mainPath,configTreeDB,cont)
                cont+=1
                cascade(mainPath2,solverDB,cont)
                #numofexperiments.append(index)
                contador += 1 # Controlar el número de experimentos que hay. Máximo 6 -> crearTab()
                
    else:
                popupmsgwarning('Number of experiments exceeded!')

        
## ****************** END OF FUNCTIONS **********************
        
#############################################################
## ******************* CLASSES ******************************

 
class KORALI(tk.Tk): #Inherited tk.tk

    # Initializing:
    def __init__(self,*args,**kwargs): # Self is implied, you don't need to pass self, but is a must.
        tk.Tk.__init__(self,*args,**kwargs)
 
        tk.Tk.wm_title(self,'KORALI')

        global contador
        global contadorVariables
        global selectedtab

        # Barra de arriba:
        self.menubar = tk.Menu(self) # Menu in the container.
        self.homemenu = tk.Menu(self.menubar, tearoff=0) # Tearoff = if clicking in the dashedline, we can make it its own window.
        self.homemenu.add_command(label='Save',command = lambda: popupmsg('Not supported just yet!'))
        self.homemenu.add_command(label='Save as...',command = lambda: popupmsg('Not supported just yet!'))
        self.homemenu.add_separator() # Separator baselr.
        self.homemenu.add_command(label='Open Project...',command = lambda: popupmsg('Not supported just yet!'))
        self.homemenu.add_command(label='Export as...',command = lambda: popupmsg('Not supported just yet!'))
        self.homemenu.add_separator() # Separator bar.
        self.homemenu.add_command(label='Exit',command = quit)
        self.menubar.add_cascade(label='Home',menu = self.homemenu)
        
        # EXPERIMENT MENU:
        self.experimentmenu = tk.Menu(self.menubar, tearoff = 1)
        self.experimentmenu.add_command(label = 'New Experiment...', command = lambda:crearTab(self,self.totalTabs))
        self.experimentmenu.add_command(label = 'New Variable...', command = lambda: crearFrameVariables())#,self.totalTabs2))
        self.experimentmenu.add_separator() # Separator baselr.
        self.experimentmenu.add_command(label = 'Delete Experiment...', command = lambda: deleteTab(self,self.totalTabs))#self.totalTabs.forget(self.tab))#lambda:deletetab(self,self.totalTabs,contador))
        self.experimentmenu.add_command(label = 'Delete Variable...', command = lambda:deleteVariable(self))
        self.experimentmenu.add_separator() # Separator baselr.
        self.experimentmenu.add_command(label = 'Done!', command = lambda:popupmsgwarning('Not done yet'))
        self.menubar.add_cascade(label = 'Experiments Configuration', menu = self.experimentmenu)

        ##### HELP MENU:
        self.helpmenu = tk.Menu(self.menubar, tearoff = 0)
        self.helpmenu.add_command(label='Tutorial', command = tutorial)
        self.menubar.add_cascade(label='Help', menu = self.helpmenu)
        

        tk.Tk.config(self, menu = self.menubar)

        # ************** TABS **********************
        # Create the TABS BAR
        self.totalTabs = ttk.Notebook(self)
        self.totalTabs.pack(expand = 1, fill = "both")

        # Call the function crearTab to create as many tabs as the user wants:
        crearTab(self,self.totalTabs)        

        
        # DOWN-TOOLBAR
        self.toolbar = tk.Frame(self, background='darkcyan')
        self.insertButt = tk.Button(self.toolbar, text='Close KORALI', command=quit)
        self.insertButt.config(bd = 2)
        self.insertButt.pack(side='right',padx=2,pady=2) # Padding options.
        self.insertButt.pack_propagate(0)
        
        self.toolbar.pack(side='bottom', fill='x')

## --------------- END OF CLASSES ------------------------    
########################################################    

configTreeDB = {}
solverDB = {}
variables = {}

readDirs(mainPath, configTreeDB,default)
readDirs(mainPath2,solverDB, default)
       
#print(variables)

app = KORALI()
app.geometry("1500x1000") # Size of our application.
app.minsize("1500","1000")
app.mainloop()
