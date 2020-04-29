import tkinter as tk
from tkinter import *
from tkinter import ttk # if python.7 Tkinter. ttk is like the CSS.
from tkinter.messagebox import showerror, showwarning, showinfo
import os, sys
import json
import copy                     # recursiveUpdate
from functools import partial # Callback to retrieve values from entries
from tkinter.filedialog import asksaveasfile, askopenfilename


# ************ VARIABLES *****************
HUGE_FONT = ("Verdana",26)
LARGE_FONT = ("Verdana",12) #Font and size.
NORM_FONT = ("Verdana",10) #Font and size.
SMALL_FONT = ("Verdana",8) #Font and size.
RES_FONT = ('Courier',12)
font = "Helvetica 10"

darkColor ='lightseagreen'
lightColor = '#00A3E0'
extraColor = '#183A54'

questions = ["Optimization","Sampling","Bayesian Inference","Hierarchical Bayesian Modeling"]
questions_seconds = ['5 Seconds','10 Seconds','30 Seconds','1 Min','10 Min','30 Min','1 Hour']

default = False # readirs function.

contador = 1 # Counts the number of experiments.
cont = 0     # Which frame are we in. Problem, Solver or Variables.

contadorVariables = 1 # Counts the number of variables.
linktoVariables = []  # Stores the information coming from Problem and Solver to update the Variable Frame.

menus = [] # Store which directories have already been read.
listofproblems = [] # Which directories are part of problem.
listofsolvers = []


experiments = {} # Global dictionary for the frames and experiments.
selectedtab = ''
selectedtab2 = ''

on = 0  #

files = [('Config Files', '*.config'),('Log Document','*.log'), 
             ('Text Document', '*.txt')] 

mainPath = '../../source/modules/problem'
mainPath2 = '../../source/modules/solver'

printVariable = 'Configuration Settings'

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
    

def explainDescription(description):
    tut = tk.Tk()
    half = len(description)/2
    tut.wm_title('Description')
    label = ttk.Label(tut, text= description[0:int(half)]+'\n'+description[int(half)::], font = NORM_FONT)
    label.pack(side='top', fill='x', pady=10)
    tut.maxsize('900','300')

    
################
################ DELETE FUNCTIONS:
def Clear(frame):
    for item in frame.winfo_children():
        item.destroy()


def clearFrame():
    global experiments
    global selectedtab
    
    frame = experiments[selectedtab]['canvas']
    frame2 = experiments[selectedtab]['rightFrame']
    frame3 = experiments[selectedtab]['bottomrightFrame']
    frame3 = frame3.tab2
    Clear(frame)
    Clear(frame2)
    Clear(frame3)

def deleteTab(totalTabs):
    global contador
    if contador >2:
        for item in totalTabs.winfo_children():
            if str(item)==totalTabs.select():
                item.destroy()
                contador -=1
                return
    else:
        popupmsgwarning('At least 2 experiments to delete 1')

def deleteVariable(self):
    global experiments
    global selectedtab
    
    totalTabs2 = experiments[selectedtab]['totalTabs2']
    #frame = experiments[selectedtab]['bottomrightFrame']
    results =  experiments[selectedtab]['results']
    ## Delete Tab and frame inside:
    if len(totalTabs2.tabs()) > 1:
        a = totalTabs2.select()
        selectedtab2Deleted = totalTabs2.tab(a, "text")
        if results[2][selectedtab2Deleted]:
             del results[2][selectedtab2Deleted]
        totalTabs2.forget(a)
        
    else:
        popupmsgwarning('At least 2 variables to delete 1')

#################
################# PRINTING FUNCTIONS:

def printdata(self,line, texto, description, fakedescription, r, c,cont):
    
    global experiments
    global selectedtab
    global selectedtab2

    results = experiments[selectedtab]['results']
    if cont == 2:
        color = 'floralwhite'
        res = results[cont][selectedtab2]
    else:
        color = 'azure'
        
        res = results[cont]
    
    double_Var = tk.DoubleVar()
    num = tk.IntVar()
    boolean = tk.BooleanVar()
    experiments[selectedtab]['results'] = results
    stringVar = {}
    self.grid_propagate(0)
    
    if 'vector' not in line:
        
        if 'string' in line or 'korali' in line: # == 'std::string
            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            stringVar[texto] = tk.StringVar()
            #self.stringVar[texto].trace('w', partial(callback,self.stringVar[texto]))
            res[texto] = tk.Entry(self,width=10,textvariable = stringVar[texto])
            res[texto].grid(row=r, column=c+1, pady= 20, padx=20, sticky = 'w')
            #print('Añado en :'+str(cont)+' esto :'+texto)
            self.corrector = self.register(validarstring)
            res[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = color ,activebackground = 'white',
                                            command = lambda:explainDescription(description))
            self.buttonLabel.grid(row=r, column=c+3,sticky='w')
            #self.label2=tk.Label(self, text = descripcion,bg='azure',relief = 'raised', font = font, width = 60)
            #self.label2.grid(row=r, column=c+3,sticky='w')
            
        elif line == 'double':
            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            res[texto] = Spinbox(self, width=10, from_=0, to=9999, wrap=True, textvariable=num, state='normal')
            res[texto].grid(row=r, column=c+1, pady= 20, padx=20)
            self.corrector = self.register(validardigit)
            res[texto].config(validate = 'key',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
            #spinbox_value = spinbox.get()
            
        elif line == 'bool':
            self.label=tk.Label(self, text=texto, width = 17,justify='left',bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            radiobutton = tk.Radiobutton(self, width=10,text='Yes' ,highlightbackground = color,activeforeground ='white', bg=color,
                                        activebackground='darkcyan',borderwidth = 0,variable = boolean, value=True)
            radiobutton.grid(row=r, column=c+1, columnspan = 1,pady= 20, padx=20, sticky = 'w')
            
            radiobutton = tk.Radiobutton(self, width=10,text='No' ,highlightbackground = color,activeforeground ='white', bg=color,
                                        activebackground='darkcyan',borderwidth = 0,variable = boolean, value=False)
            radiobutton.grid(row=r, column=c+2, columnspan = 1,pady= 20, padx=20, sticky = 'w')
            res[texto] = boolean
             
        elif line == 'size_t':
            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            stringVar[texto] = tk.StringVar()
            res[texto] = tk.Entry(self,width=10,textvariable = stringVar[texto])
            res[texto].grid(row=r, column=c+1, pady= 20, padx=20, sticky = 'w')
            self.corrector = self.register(validarstring)
            res[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = color,activebackground = 'white',
                                            command = lambda:explainDescription(description))
            self.buttonLabel.grid(row=r, column=c+3,sticky='w')
            
            
    else:
        if line =="std::vector<size_t>":
            
            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            stringVar[texto] = tk.StringVar()
            res[texto] = tk.Entry(self,width=10,textvariable = stringVar[texto])
            res[texto].grid(row=r, column=c+1, pady= 20, padx=20, sticky = 'w')
            self.corrector = self.register(validarstring)
            res[texto].config(validate = 'focus',validatecommand = (self.corrector,'%P')) # %P represents the parameter we want to pass to validate.
            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = color,activebackground = 'white',
                                            command = lambda:explainDescription(description))
            self.buttonLabel.grid(row=r, column=c+3,sticky='w')
        
        if 'double' in line:
            self.label=tk.Label(self, text=texto, width = 17,justify='left', bg=color, anchor="w", font="Arial 16")
            self.label.grid(row=r, column=c,pady = 20,padx=20, sticky='w')
            stringVar[texto] = tk.StringVar()
            res[texto] = tk.Entry(self,width=10,textvariable = stringVar[texto])
            res[texto].grid(row=r, column=1, pady= 20, padx=20)
            self.corrector = self.register(validardigit)
            self.buttonLabel = tk.Button(self, text = fakedescription, width = 45, activeforeground = 'darkcyan', bg = color,activebackground = 'white',
                                            command = lambda:explainDescription(description))
            self.buttonLabel.grid(row=r, column=c+3,sticky='w')

def printVariables(self,directorio,DB,cont):
    '''
    linktoVariables refers to the .config path of each element of the cascade. This path also works
    as a key in 'variables' dictionary, and as a value, we can find their variables configuration.
    '''
    global on
    global variables
    global linktoVariables
    global linktoVariables0
    global linktoVariables1
    global experiments
    global selectedtab2
    global selectedtab

    self = experiments[selectedtab]['bottomrightFrame']
    self = self.tab2
    
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

    linktoVariables = []
    if on == 0:
        if linktoVariables0:
            linktoVariables.append(linktoVariables0)    # Añadimos las variables que vienen del Problem.
        if linktoVariables1:
            linktoVariables.append(linktoVariables1)    # Añadimos las variables que vienen del Solver.
        cont = 2
        # Clean the frame:
        for widget in self.winfo_children():
            widget.destroy()
        results = experiments[selectedtab]['results']
        # Coger nombre variable:
        results[cont] = {}
        results[cont][selectedtab2] = {}          
        ro = 0
        co = 0
        for part in linktoVariables:
            for llave in variables[part]:
                for var in llave.keys():
                    if var == 'Type':
                        line = llave['Type']
                        # llave son las llaves del diccionario, por ejemplo: Type, Function, Description, Produced By...
                        texto = llave['Name']
                        texto = texto[0] # Remove the '{}' from the label name.
                        description = llave['Description']
                        if len(description)>45:
                            fakedescription = str(description[0:42])+'...'
                        else:
                            fakedescription = description
                        printdata(self,line, texto, description,fakedescription, ro, co, cont)
                        ro+=1
    '''
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

                    for widget in self.winfo_children():
                        print(widget)
                        widget.destroy()
                    printdata(self,variables[part][llave], texto, description,fakedescription, ro, co, cont)
                ro+=1
        '''
    

def printConfig(directorio,DB, cont):
    
    global selectedtab
    global experiments

    # From which cascade do we come from? Problem, Solver or Variable?
    if cont == 0:
        self = experiments[selectedtab]['rightFrame']
    elif cont == 1:
        self = experiments[selectedtab]['canvas']
    #elif cont == 2:
    #    self = experiments[selectedtab]['bottomrightFrame']
    #####

    ## DELETE ANY WIDGET THAT WAS THERE BEFORE:
    for widget in self.winfo_children():
        widget.destroy()
    
    # Set variables:
    r = 0
    c = 0
    results = experiments[selectedtab]['results']
    results[cont]={}

    # Start reading and printing on screen: # DON'T DELETE THIS COMMENT -> TITULO FRAMES : list(DB.keys())[0]
    self.titulo=tk.Label(self, text=directorio,justify='left', anchor="w", font="Arial 18", bg = 'darkcyan', fg='white')
    self.titulo.grid(row=r, column=c,pady = 10 ,padx=20, sticky='w', columnspan = 2)
    config = DB[directorio]['herencia']
    conf_sett = 'Configuration Settings'
    
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

                                    fakedescription = description
                                    printdata(self,dicc[key2], texto, description,fakedescription, r, c,cont)
                                r+=1
    else:
        popupmsgwarning('No Configuration Settings found')
    '''
    if cont == 1:
                bottomleftFrame = experiments[selectedtab]['bottomleftFrame']
                canvas = experiments[selectedtab]['canvas']
                
                hbar=Scrollbar(bottomleftFrame,orient=HORIZONTAL)
                hbar.pack(side=BOTTOM,fill=X)
                hbar.config(command=canvas.xview)
                vbar=Scrollbar(bottomleftFrame,orient=VERTICAL)
                vbar.pack(side=RIGHT,fill=Y)
                vbar.config(command=canvas.yview)
                canvas.config(width=950,height=850)
                canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
                canvas.pack(side=LEFT,expand=True,fill=BOTH)
                canvas.pack_propagate(0)
    '''
                                    
##########################
########################## MAIN FUNCTIONS:
def importFile():
    global experiments
    global configTreeDB
    global solverDB
    global files

    filename = askopenfilename(filetypes=files)
    print(filename)
    parsedFile = ''
    try:
        file = open(filename,'r')
        parsedFile = json.load(file)
    except:
        popupmsgwarning('File is not readable')
    finally:
        file.close()
    bottomrightFrame = experiments[selectedtab]['bottomrightFrame']
    if parsedFile != '':
        
        valuesProblem = []
        valuesSolver = []
        valuesVariables = {}
        rightFrame = experiments[selectedtab]['rightFrame']
        results = experiments[selectedtab]['results']
        
        for directorio in parsedFile.keys(): # Loop through the keys of the config file.
            if directorio in listofproblems:
                cont = 0
                printConfig(directorio,configTreeDB,cont)
                printVariables(bottomrightFrame.tab2,directorio,configTreeDB,cont)
                for key2 in parsedFile[directorio]:
                    valuesProblem.append(parsedFile[directorio][key2])
                numofentry = 0
                for key in results[cont].keys():
                    e = results[cont][key]
                    e.insert(0,valuesProblem[numofentry])
                    numofentry+=1
            elif directorio in listofsolvers:
                cont = 1
                printConfig(directorio,solverDB,cont)
                printVariables(bottomrightFrame.tab2,directorio,solverDB,cont)
                for key2 in parsedFile[directorio]:
                    valuesSolver.append(parsedFile[directorio][key2])
                numofentry = 0
                for key in results[cont].keys():
                    e = results[cont][key]
                    if isinstance(e,BooleanVar):
                        value = valuesSolver[numofentry]
                        e.set(value)
                    else:
                        default = e.get()
                        if default == '0':
                            e.delete(0,tk.END)
                            e.insert(END,valuesSolver[numofentry])
                        else:
                            e.insert(END,valuesSolver[numofentry])
                    numofentry+=1
            elif 'variabl' in directorio or 'Variabl' in directorio:
                cont = 2
                numofentry = 0
                if len(parsedFile[directorio].keys()) >1:
                    v = 0 # If there are several Variables Tabs in the imported file...
                    for variableTab in parsedFile[directorio].keys():
                        if v>0:
                            crearFrameVariables()
                            v+=1
                        valuesVariables[variableTab] = []
                        line = parsedFile[directorio][variableTab]
                        v+=1
                        for dictionary in line:
                            for key in dictionary:
                                valuesVariables[variableTab].append(dictionary[key])
                else:
                    for variableTab in parsedFile[directorio].keys():
                        valuesVariables[variableTab] = []
                        line = parsedFile[directorio][variableTab]
                        for dictionary in line:
                            for key in dictionary:
                                valuesVariables[variableTab].append(dictionary[key])                       
                for key in results[cont].keys():
                    for varkey in valuesVariables.keys():
                            if key == varkey:
                                numofentry = 0
                                for key2 in results[cont][key]:
                                    e = results[cont][key][key2]
                                    e.insert(0,valuesVariables[varkey][numofentry])
                                    numofentry+=1
    else:
        popupmsgwarning('Could not import the file')
            
        
        '''
        for frame in parsedFile.keys():
            for texto in parsedFile[frame]:
                w = parsedFile[frame][texto]
                val = w.get()
                results2[frame][texto] = valprint(valuesProblem, valuesSolver)
        '''
           
def createConfig():
    global selectedtab # To know at which experiment we are.
    global experiments
    global files # Types of files.

    results = experiments[selectedtab]['results']
    ### Avoiding that the user clicks create.config without choosing a problem/solver
    if 2 not in results.keys():
        popupmsgwarning('First choose a problem and a solver')
        return
    
    #results = experiments[selectedtab]['results']
    bottomleftFr = experiments[selectedtab]['canvas']
    rightFr = experiments[selectedtab]['rightFrame']
    variableFr = experiments[selectedtab]['totalTabs2']
##    print('BEFOOOOOOOOOOOOOOOOORE',results2.keys(),'\n')
##    results2 = copy.deepcopy(results)
    results2 = {}
    for key in results:
         results2[key] = results[key].copy()
    results2[rightFr.titulo.cget('text')] = results2.pop(0)
    results2[bottomleftFr.titulo.cget('text')] = results2.pop(1)
    results2['Variables'] = results2.pop(2)
    
    for frame in results2.keys():
        for texto in results2[frame]:
            w = results2[frame][texto]
            if isinstance(w,dict):
                results2[frame][texto] = []
                for key in w:
                    val = w[key].get()
                    d = {key:val}
                    results2[frame][texto].append(d)
##                    print(results2[frame][texto])
            else:
                val = w.get()
                results2[frame][texto] = val
##    cwd = os.getcwd()
##    file = open(cwd+"/filename.config", "w")
    #file.write('### .config File obtained from Korali### \n \n'+selectedtab+' {\n')

    #### ASK TO DOWNLOAD AND SAVE THE FILE:
    filesaved = asksaveasfile(filetypes = files, defaultextension = files)
    json.dump(results2, filesaved, indent = 3)
    filesaved.close()

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

def readDirs(filePath,DB,default):
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
                    DB[dirName] = dirInfoDic
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
                            if '****\n' in line or '===\n' in line:
                                name = lines[i+1]
                                dirInfoDic['names'] = name.replace('\n','') # Quitar '\n'
                                break
                            else:
                                pass
                    finally:
                        file.close() ## Cierra el archivo README.
            continue
        dirs=splitPath(dirPath)
        levels=len(dirs)
        if levels > sublevel:
            del dirNames[:]
            continue
        readDirs(dirPath, DB, dest)
        dirName = dirs[levels - 1]
        childrenList += [dirName]
    crearVariables(DB)
    
### CREAR EL DICCIONARIO DE VARIABLES COMBINANDO LOS DOS DICCIONARIOS ANTERIORES
def crearVariables(DB):
    for key in DB.keys():
        if 'Variables Configuration' in DB[key]['herencia'].keys():
            if DB[key]['herencia']['Variables Configuration']:
                variables[DB[key]['config']] = DB[key]['herencia']['Variables Configuration']
            else:
                variables[DB[key]['config']] = {}
        else:
            variables[DB[key]['config']] = {}            
             ######################   
                    
def crearMenu(padre,directorio,DB,cont):
    global menus
    global experiments
    global listofproblems
    global listofsolvers
    
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
    if cont == 0:
        if directorio not in listofproblems:
            listofproblems.append(directorio)
    elif cont == 1:
        if directorio not in listofsolvers:
            listofsolvers.append(directorio)

def select_item():
    pass
def remove_item():
    pass
def add_item():
    pass

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
    global linktoVariables
    global variables
    global selectedtab2

##    print('first:',selectedtab2)
    
    bottomrightFrame = experiments[selectedtab]['bottomrightFrame']
    totalTabs2 = experiments[selectedtab]['totalTabs2']
    if len(totalTabs2.tabs()) == 0:
        bottomrightFrame.tab2 = tk.Frame(bottomrightFrame, height = 450, width = 300, background = 'floralwhite')#florawhite
        selectedtab2 = 'Variable '+str(contadorVariables)
        totalTabs2.add(bottomrightFrame.tab2, text = selectedtab2)
        contadorVariables  += 1
    #for item in self.totalTabs
    else:
        if linktoVariables == []:
            popupmsgwarning('First choose a Problem and a Solver')
        else:
            if contadorVariables < 15 :
                    ## Create Tab and frames inside:
                    bottomrightFrame.tab2 = tk.Frame(bottomrightFrame, height = 450, width = 300, background = 'floralwhite')#florawhite
                    selectedtab2 = 'Variable '+str(contadorVariables)
                    totalTabs2.add(bottomrightFrame.tab2, text = selectedtab2)
                    self = bottomrightFrame.tab2
                    cont = 2
                    results = experiments[selectedtab]['results']
                    # Coger nombre variable:
                    #variableFr = experiments[selectedtab]['totalTabs2']
                    results[cont][selectedtab2] = {}
                    ro = 0
                    co = 0
                    for widget in self.winfo_children():
                        widget.destroy()
                    for part in linktoVariables:
##                        print('This is the part:',part,'\n')
                        for llave in variables[part]:
##                            print('This is the llave:',llave,'\n')
                            for var in llave.keys():
##                                print('This is the var:',var,'\n')
                                if var == 'Type':
                                    getname=llave['Name']
                                    getdesc=llave['Description']
                                    line = llave['Type']
                                    # llave son las llaves del diccionario, por ejemplo: Type, Function, Description, Produced By...
                                    texto = getname
                                    texto = texto[0] # Remove the '{}' from the label name.
                                    description = getdesc
                                    if len(description)>45:
                                        fakedescription = str(description[0:42])+'...'
                                    else:
                                        fakedescription = description
                                    printdata(self,line, texto, description,fakedescription, ro, co, cont)
                            ro+=1
                    ## IT ALREADY ADDS THE NEW VARIABLE TO THE DICTIONARY OF RESULTS BY USING THE FUNCTION PRNTDATA.
            else:
                popupmsgwarning('Number of Variables exceeded!')
            contadorVariables+=1
            
## Crea los tabs de uno en uno
def crearTab(self,totalTabs):
    global contador
    global selectedtab
    global experiments
    global linktoVariables
        
    linktoVariables = []

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
                self.button_done = tk.Button(self.leftFrame, text = 'Create .config file',command = lambda: createConfig(),
                                             activebackground = 'darkcyan',activeforeground='azure', bd = 3, bg = 'teal',fg = 'azure', font = 'Arial 12',
                                             width = 20)

                self.button_done.grid(row=0, column=3, pady= 20, padx=20, sticky = 'E')
                
                self.button_Import = tk.Button(self.leftFrame, text = 'Import .config file',command = lambda: importFile(),
                                             activebackground = 'darkcyan',activeforeground='azure', bd = 3, bg = 'teal',fg = 'azure', font = 'Arial 12',
                                             width = 20)

                self.button_Import.grid(row=0, column=4, pady= 20, padx=20, sticky = 'E')
                self.update_idletasks


                self.rightFrame = tk.Frame(self.tab, width = 900, height = 450,background ='azure', borderwidth = 1,relief = 'groove')
                self.rightFrame.grid(column = 1, row = 0, stick = 'nsew',rowspan = 2) #, sticky = 'ne', rowspan = 1, columnspan = 1)


                self.bottomleftFrame1 = tk.Frame(self.tab, background = 'azure', width = 950, height = 80, borderwidth = 3,relief = 'groove')#width = 950, height = 850
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
                
                self.canvas=Canvas(self.bottomleftFrame1,width = 950, height = 840,bg = 'azure', scrollregion= (0,0,1500,1500)) #yscrollcommand = vbar.set
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
                
                #self.canvas.pack_propagate(0)
                
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
                experiments[selectedtab]['results'] = {}
                
                expElements['leftFrame'] = self.leftFrame
                expElements['rightFrame'] = self.rightFrame
                expElements['bottomrightFrame'] = self.bottomrightFrame
                expElements['bottomleftFrame'] = self.bottomleftFrame1
                expElements['canvas'] = self.canvas

                self.totalTabs2 = ttk.Notebook(self.bottomrightFrame)
                self.totalTabs2.pack(expand = 1, fill = "both")
               
                expElements['totalTabs2'] = self.totalTabs2

                crearFrameVariables()

                self.leftFrame.grid_propagate(0)
                self.rightFrame.grid_propagate(0)
                #self.bottomleftFrame.grid_propagate(0)
                self.bottomrightFrame.grid_propagate(0)
                self.bottomleftFrame1.grid_propagate(0)
                

               # LEFT SIDE OF TAB1:
                
                ## CASCADE:
                # Elegir el título de los Menús:
                cont= 0
                cascade(mainPath,configTreeDB,cont)
                cont+=1
                cascade(mainPath2,solverDB,cont)
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

        global selectedtab


        container = tk.Frame(self) # Edge of the window.
        container.pack(side = "top", fill = "both", expand = True) # Fill the entire space. Expand = If there is white space, you can expand it.
##        container.grid_rowconfigure(0, weight=1) # The minimum is 0. weight is the priority... who goes first...
##        container.grid_columnconfigure(0, weight=1)

        
        # Barra de arriba:
        self.menubar = tk.Menu(self) # Menu in the container.
        self.homemenu = tk.Menu(self.menubar, tearoff=0) # Tearoff = if clicking in the dashedline, we can make it its own window.
        self.homemenu.add_command(label='Experiments',command = lambda: self.show_frame(KORALI_Page))
        self.homemenu.add_command(label='Second Part',command = lambda: self.show_frame(KORALI_Graph))
        self.homemenu.add_separator()
        self.homemenu.add_command(label='Save',command = lambda: popupmsgwarning('SAVE'))
        self.homemenu.add_command(label='Save as...',command = lambda: createConfig())
        self.homemenu.add_separator() # Separator baselr.
        self.homemenu.add_command(label='Import File...',command = lambda: importFile())
        self.homemenu.add_command(label='Export as...',command = lambda: createConfig())
        self.homemenu.add_separator() # Separator bar.
        self.homemenu.add_command(label='Exit',command = quit)
        self.menubar.add_cascade(label='Home',menu = self.homemenu)
        
        # EXPERIMENT MENU:
        self.experimentmenu = tk.Menu(self.menubar, tearoff = 1)
        self.experimentmenu.add_command(label = 'New Experiment...', command = lambda:crearTab(self,self.totalTabs))
        self.experimentmenu.add_command(label = 'New Variable...', command = lambda: crearFrameVariables())#,self.totalTabs2))
        self.experimentmenu.add_separator() # Separator baselr.
        self.experimentmenu.add_command(label = 'Delete Experiment...', command = lambda: deleteTab(self.totalTabs))#self.totalTabs.forget(self.tab))#lambda:deletetab(self,self.totalTabs,contador))
        self.experimentmenu.add_command(label = 'Delete Variable...', command = lambda:deleteVariable(self))
        self.experimentmenu.add_separator() # Separator baselr.
        self.experimentmenu.add_command(label = 'Clear Experiment...', command = lambda:clearFrame())
        self.menubar.add_cascade(label = 'Experiments Configuration', menu = self.experimentmenu)

        ##### HELP MENU:
        self.helpmenu = tk.Menu(self.menubar, tearoff = 0)
        self.helpmenu.add_command(label='Tutorial', command = tutorial)
        self.menubar.add_cascade(label='Help', menu = self.helpmenu)
        

        tk.Tk.config(self, menu = self.menubar)

        # ************** TABS **********************
        # Create the TABS BAR
##        self.totalTabs = ttk.Notebook(self)
##        self.totalTabs.pack(expand = 1, fill = "both")

        # Call the function crearTab to create as many tabs as the user wants:       
##        crearTab(self,self.totalTabs)

        ####################################
        self.frames = {} # Dictionary.

        for F in (StartPage,KORALI_Page, KORALI_Graph): # List of Pages.

            frame = F(container,self) #Pass through container, and self.
            self.frames[F] = frame 
            frame.grid(row=0, column=0, sticky='nsew') # Sticky = Strech everything to north,south,e,w..

        self.show_frame(StartPage)

    def show_frame(self,cont): # Cont = controler.
        frame = self.frames[cont]
        frame.tkraise() # Raise it to the front!

        ###################################       
        # DOWN-TOOLBAR
        self.toolbar = tk.Frame(self, background='darkcyan')
        
        self.insertButt = tk.Button(self.toolbar, text='Close KORALI', command=quit)
        self.insertButt.config(bd = 2)
        self.insertButt.pack(side='right',padx=2,pady=2) # Padding options.
        self.insertButt.pack_propagate(0)
        
        self.toolbar.pack(side='bottom', fill='x')
        #self.toolbar.pack_propagate(0)

class KORALI_Page(tk.Frame): # Page with Graphs.
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)

        # Create the TABS BAR
        self.totalTabs = ttk.Notebook(self)
        self.totalTabs.pack(expand = 1, fill = "both")

        # Call the function crearTab to create as many tabs as the user wants:       
        crearTab(self,self.totalTabs)

class KORALI_Graph(tk.Frame): # Page with Graphs.
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)


        button2 = tk.Label(self, text = "This is the frame for the next part", width = 150, height = 10, relief=tk.RAISED)
        button2.pack(padx = 1, pady = 50 )



class StartPage(tk.Frame): # inherit everything from tk.Frame
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent) #Parent = KORALI
        #tk.Frame.config(self,bg='aliceblue') # COLORES
        # Button to allow the user to navigate through windows:
        button1 = tk.Button(self, text = "Welcome to KORALI", font = HUGE_FONT,
                                command = lambda: controller.show_frame(KORALI_Page)) # lambda creates a quick thronaway function.
        button1.config(borderwidth = 2,activeforeground = 'darkcyan',relief = 'groove')
        button1.pack(side = 'top', pady=400)

    

## --------------- END OF CLASSES ------------------------    
########################################################    

configTreeDB = {}
solverDB = {}
variables = {}

readDirs(mainPath, configTreeDB,default)
readDirs(mainPath2,solverDB, default)
#print(configTreeDB)
       
app = KORALI()
app.geometry("1500x1000") # Size of our application.
app.minsize("1500","1000")
app.mainloop()
