import os
import json

# Read result files
def getResultFiles(src, start=None, end=None, noisy=False):
    if not os.path.exists(src):
     return []
    
    resultfilesTmp = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    resultfilesTmp = sorted(resultfilesTmp)

    if ('initial.json' not in resultfilesTmp):
        #print("[Korali] Error: Did not find file 'initial.json' in the result folder...")
        return []
    
    resultfilesTmp.remove('initial.json')
    if ('final.json' in resultfilesTmp): 
        resultfilesTmp.remove('final.json')

    if (resultfilesTmp == []):
        return []

    initialpath = '{0}/initial.json'.format(src)
    runId = None
    with open(initialpath) as f:
        dataString = f.read()
        dataString = dataString.replace('+INFINITY', '1.0e+300').replace('-INFINITY', '-1.0e+300').replace('NaN', '-1.0e+300')
        data  = json.loads(dataString)
        runId = data['General']['Run ID']

    resultfiles = [] # Init Return Value
    for filename in resultfilesTmp:
        path   = '{0}/{1}'.format(src, filename)
        
        with open(path) as f:
            dataString = f.read()
            dataString = dataString.replace('+INFINITY', '1.0e+300').replace('-INFINITY', '-1.0e+300').replace('NaN', '-1.0e+300')
            data  = json.loads(dataString)
            gen  = data['General']['Current Generation']
            
            if verifyResultFile(data, path, runId, start, end, noisy):
                resultfiles.append(filename)
    
    return resultfiles


# Open file and verify runId and current generation in [start, end]
def verifyResultFile(data, path, runId, start=None, end=None, noisy=True): 
    currentGeneration = data['General']['Current Generation']

    if ( (start is not None) and (currentGeneration < start)):
        return False

    if ( (end is not None) and (currentGeneration > end)):
        return False

    if (data['General']['Run ID'] != runId):
       
        if(noisy == True):
            print("[Korali] Warning: Skipping file {0}, results origin from a different experiment (different runid)".format(path))
        return False

 
    return True


# Check generation greater equal lowerBound
def verifyResultGeneration(generation, lowerBound):
    if (generation is not None and generation < lowerBound):
        print("[Korali] GENERATION must be greater equal "\
                "{0}.".format(str(lowerBound)))
        exit(-1)

