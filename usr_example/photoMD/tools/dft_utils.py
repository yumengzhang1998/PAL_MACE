#!/usr/bin/env python
import os, sys, resource, time, random, subprocess, shutil, uuid, getpass, socket, yaml, re
import numpy as np
import scipy.spatial as scsp
try:
    from StringIO import StringIO as mStringIO
except ImportError:
    from io import StringIO as mStringIO


kcal_to_eV=0.0433641153
kB=8.6173303e-5 #eV/K
T=298.15
kBT=kB*T
AToBohr=1.889725989
HToeV = 27.211399


def readXYZ(filename):
    infile=open(filename,"r")
    coords=[]
    elements=[]
    lines=infile.readlines()
    if len(lines)<3:
        exit("ERROR: no coordinates found in %s/%s"%(os.getcwd(), filename))
    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    coords=np.array(coords)
    return coords,elements

def sci_to_float(s):
    """convert string with scientific notation to float"""
    l = re.split('[a-zA-Z]', s)
    if len(l) == 1:
        return float(l[0])
    elif len(l) == 2:
        return float(l[0]) * (10.0**int(l[1]))
    else:
        return None

#############
# Turbomole #
#############

def copy_dir_contents_to_dir(in_directory, out_directory, dir_to_copy):
    try:
        #We copy all files in in_directory to out_directory without following directories recursively:
        file_with_dir = "%s/%s" % (in_directory, dir_to_copy)
        if os.path.exists(file_with_dir):
            shutil.copytree(file_with_dir, "%s/%s"%(out_directory,dir_to_copy))
        else:
            print("did not find the directory %s to copy back from scratch"%(file_with_dir))
            exit()
    except Exception as exc:
        print("Moving files to from %s to %s has failed. Reraising Exception:" % (in_directory, out_directory))
        print(exc)
        raise



def GoToScratch():
    oldcwd = os.getcwd()
    try:
        SCRATCH_BASE = os.environ["SCRATCH"]
        username = getpass.getuser()
        randstring = uuid.uuid4()
        scratch_directory = "%s/%s/%s" % (SCRATCH_BASE, username, randstring)
        os.makedirs(scratch_directory)
    except KeyError as exc:
        print("A KeyError occured, when querying the Scratch Directory. Check the environment settings. Exception was: %s. Turning off scratch handling." % ( exc ))
        scratch_directory=oldcwd
    except Exception as exc:
        #In case there was something unforseen, we reraise to bomb out of the application.
        print("An unexpected exception as occured of type %s. Exception was: %s. Reraising." % (type(exc), exc))
        raise
    os.chdir(scratch_directory)
    return([oldcwd, scratch_directory])


def ComeBachFromScratch(oldcwd, scratch_directory, dir_to_copy):
    if oldcwd != scratch_directory:
        #copy result back to oldcwd, change back, remove scratch
        copy_dir_contents_to_dir(scratch_directory, oldcwd, dir_to_copy)
        os.chdir(oldcwd)
        shutil.rmtree(scratch_directory)
    else:
        print("Warning, oldcwd ", oldcwd, "was equal to scratch_directory", scratch_directory, "review log for exceptions.")



def dft_calc(dft_settings, coords, elements, opt=False, grad=False, hess=False, charge=0, freeze=[]):

    if opt and grad:
        exit("opt and grad are exclusive")
    if hess and grad:
        exit("hess and grad are exclusive")

    if hess or grad:
        if len(freeze)!=0:
            print("WARNING: please test the combination of hess/grad and freeze carefully")

    rundir="dft_tmpdir_%s"%(uuid.uuid4())
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    else:
        if len(os.listdir(rundir))>0:
            os.system("rm %s/*"%(rundir))

    startdir=os.getcwd()
    os.chdir(rundir)

    PrepTMInputNormal(".", coords, elements)
    RunTMCalculation(".", dft_settings)

    if opt:
        os.system("t2x coord > opt.xyz")
        coords_new, elements_new = readXYZ("opt.xyz")
    else:
        coords_new, elements_new = None, None

    if grad:
        grad = read_dft_grad()
    else:
        grad = None

    if hess:
        hess, vibspectrum, reduced_masses = read_dft_hess()
    else:
        hess, vibspectrum, reduced_masses = None, None, None

    e = getTMEnergies(".")[2]

    os.chdir(startdir)

    #os.system("rm -r %s"%(rundir))

    results={"energy": e, "coords": coords_new, "elements": elements_new, "gradient": grad, "hessian": hess, "vibspectrum": vibspectrum, "reduced_masses": reduced_masses}
    return(results)


def read_dft_grad():
    if not os.path.exists("gradient"):
        return(None)
    grad = []
    for line in open("gradient","r"):
        if len(line.split())==3 and "grad" not in line:
            line = line.replace("D","E")
            grad.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
    if len(grad)==0:
        grad=None
    else:
        grad = np.array(grad)*HToeV*AToBohr
    return(grad)


def read_dft_hess():
    hess = None
    if not os.path.exists("hessian"):
        return(None, None, None)
    hess = []
    for line in open("hessian","r"):
        if "hess" not in line:
            for x in line.split():
                hess.append(float(x))
    if len(hess)==0:
        hess=None
    else:
        hess = np.array(hess)

    vibspectrum = None
    if not os.path.exists("vibspectrum"):
        return(None, None, None)
    vibspectrum = []
    read=False
    for line in open("vibspectrum","r"):
        if "end" in line:
            read=False

        if read:
            if len(line.split())==5:
                vibspectrum.append(float(line.split()[1]))
            elif len(line.split())==6:
                vibspectrum.append(float(line.split()[2]))
            else:
                print("WARNING: weird line length: %s"%(line))
        if "RAMAN" in line:
            read=True
    
    reduced_masses = None
    if not os.path.exists("g98.out"):
        print("g98.out not found")
        return(None, None, None)
    reduced_masses = []
    read=False
    for line in open("g98.out","r"):
        if "Red. masses" in line:
            for x in line.split()[3:]:
                try:
                    reduced_masses.append(float(x))
                except:
                    pass

    if len(vibspectrum)==0:
        vibspectrum=None
        print("no vibspectrum found")
    else:
        vibspectrum = np.array(vibspectrum)

    if len(reduced_masses)==0:
        reduced_masses = None
        print("no reduced masses found")
    else:
        reduced_masses = np.array(reduced_masses)

    return(hess, vibspectrum, reduced_masses)





def RunTMCalculation(moldir, dft_settings):
    startdir=os.getcwd()
    os.chdir(moldir)

    instring=prep_define_file(dft_settings, 0)
    ExecuteDefineString(instring)

    if dft_settings["copy_mos"]:
        if os.path.exists("%s/pre_optimization/mos"%(dft_settings["main_directory"])):
            print("   ---   Copy the old mos file from precalculation")
            os.system("cp %s/pre_optimization/mos ."%(dft_settings["main_directory"]))
        else:
            print("WARNING: Did not find old mos file in %s/pre_optimization"%(dft_settings["main_directory"]))
            
    if dft_settings["copy_control"]:
        if os.path.exists("%s/pre_optimization/control"%(dft_settings["main_directory"])):
            print("   ---   Copy the old control file from precalculation")
            os.system("rm control")
            os.system("cp %s/pre_optimization/control ."%(dft_settings["main_directory"]))
        else:
            print("WARNING: Did not find old control file in %s/pre_optimization"%(dft_settings["main_directory"]))

    if dft_settings["use_dispersions"]:
        AddStatementToControl("control", "$disp3")

    if dft_settings["turbomole_method"]=="ridft":
        os.system("ridft > TM.out")
        os.system("rdgrad > rdgrad.out")
    elif dft_settings["turbomole_method"]=="dscf":
        os.system("dscf > TM.out")
        os.system("rdgrad > rdgrad.out")
    elif dft_settings["turbomole_method"]=="escf":
        os.system("escf > TM.out")
        os.system("egrad > egrad.out")
    else:
        exit("ERROR in turbomole_method: %s"%(dft_settings["turbomole_method"]))

    finished=False   
    number_of_iterations=None
    for line in open("TM.out","r"):
        if "convergence criteria satisfied after" in line:
            number_of_iterations=int(line.split()[4])
        if "all done" in line:
            finished=True
            break
    if number_of_iterations!=None:
        print("   ---   converged after %i iterations"%(number_of_iterations))
    else:
        pass

    if finished:
        os.system("eiger > eiger.out")

    os.chdir(startdir)
    return(finished)



def RunTMRelaxation(moldir,dft_settings):
    startdir=os.getcwd()
    os.chdir(moldir)

    instring=prep_define_file(dft_settings, 0)
    ExecuteDefineString(instring)

    if dft_settings["use_dispersions"]:
        AddStatementToControl("control", "$disp3")

    if dft_settings["turbomole_method"]=="ridft":
        os.system("jobex -ri -c 200 > jobex.out")
    elif dft_settings["turbomole_method"]=="dscf":
        os.system("jobex -c 200 > jobex.out")
    else:
        exit("ERROR in turbomole_method: %s"%(dft_settings["turbomole_method"]))
    
    if os.path.exists("GEO_OPT_CONVERGED"):
        finished=True
        os.system("eiger > eiger.out")
    else:
        finished=False

    AddStatementToControl("control", "$esp_fit kollman")
    if dft_settings["turbomole_method"]=="ridft":
        os.system("ridft -proper > TM_proper.out")
    elif dft_settings["turbomole_method"]=="dscf":
        os.system("dscf -proper > TM_proper.out")

    os.chdir(startdir)
    return(finished)

def getTMpartialcharges(outfilename,noOfAtoms):
    TMfile=open(outfilename,"r")
    lines=TMfile.readlines()
    TMfile.close()
    #finding position of ESP partial charges in file
    idx=0
    for line in lines:
        spl=line.split()
        if len(spl)!=0 and len(spl)!=1:
            if spl[0]=="atom" and spl[1]=="radius/au":
                index=idx+1
                break
        idx+=1        
    partialCharges=[]
    for i in range(noOfAtoms):
        partialCharges.append( float(lines[idx+1+i].split()[3]) )
    return(partialCharges)

def getTMCoordinates(moldir, startOrEnd):
    
    infile=open("%s/gradient"%(moldir))
    lines=infile.readlines()
    infile.close()
    coordinates_all=[]
    eles=[]
    for idx,line in enumerate(lines):
        if "cycle =      2" in line:
            number_of_atoms=(idx-2)/2
            break
    print("number of atoms: %i"%(number_of_atoms))

    if len(lines)% (number_of_atoms*2+1) !=2:
        print("WARNING")
        exit()

    number_of_steps=(len(lines)-2)/ (number_of_atoms*2+1)
    print("found %i gradient steps"%(number_of_steps))

    for step in range(0,number_of_steps):
        coordinates_all.append([])
        eles.append([])
        startline=1+step*(number_of_atoms*2+1)+1
        for line in lines[startline:startline+number_of_atoms]:
            coordinates_all[step].append([float(line.split()[0])/AToBohr,float(line.split()[1])/AToBohr,float(line.split()[2])/AToBohr])
            eles[step].append(line.split()[3])


    if startOrEnd =="end":
        coordinates=coordinates_all[-1]
        elements=eles[-1]
    elif startOrEnd=="start":
        coordinates=coordinates_all[0]
        elements=eles[0]

    return(coordinates, elements)



def PrepTMInputNormal(moldir, coords, elements):
    coordfile = open("%s/coord"%(moldir), 'w')
    coordfile.write("$coord \n")
    for idx, atom in enumerate(coords):
        coordfile.write("%f  %f  %f  %s\n" % (atom[0] * AToBohr, atom[1] * AToBohr, atom[2] * AToBohr, elements[idx]))
    coordfile.write("$end\n")
    coordfile.close()
    return ()


def PrepTMInput(moldir,coords,elements,dihedral,dft_settings):

    frozen_atoms=dihedral[0]
    coordfile = open("%s/coord"%(moldir), 'w')
    coordfile.write("$coord \n")
    for idx, atom in enumerate(coords):
        if idx in frozen_atoms:
            frozen = "  f"
        else:
            frozen = ""
        coordfile.write("%f  %f  %f  %s%s\n" % (atom[0] * AToBohr, atom[1] * AToBohr, atom[2] * AToBohr, elements[idx], frozen))
    coordfile.write("$end\n")
    coordfile.close()

    return()


def getTMEnergies(moldir):
    eigerfile=open("%s/eiger.out"%(moldir),"r")
    eigerlines=eigerfile.readlines()
    eigerfile.close()
    total_energy=0.0
    energy_homo=0.0
    energy_lumo=0.0
    for eigerline in eigerlines:
        if len(eigerline.split())!=0:
            if eigerline.split()[0]=="Total":
                total_energy=eigerline.split()[3] # hartree energy, if eV wanted, use index 6
            elif eigerline.split()[0]=="HOMO:":
                energy_homo=eigerline.split()[7]
            elif eigerline.split()[0]=="LUMO:":
                energy_lumo=eigerline.split()[7]
                break
    return([float(energy_homo),float(energy_lumo),float(total_energy)])




def ExecuteDefineString(instring):
    instring = instring + "\n\n\n\n"
    out = ""
    err = ""

    process = subprocess.Popen(["define"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=setulimit, encoding='utf8')
    out, err = process.communicate(input = instring)
    if "normally" in err.split():
        return
    if "normally" not in err.split():
        print("ERROR in define")
        print("STDOUT was: %s"%(out))
        print("STDERR was: %s"%(err))
        print("Now printing define input:")
        print("--------------------------")
        print(instring)
        print("--------------------------")
        with open("define.input",'w') as defineinput:
            defineinput.write(instring)
        exit()

def setulimit():
    resource.setrlimit(resource.RLIMIT_STACK,(-1,resource.RLIM_INFINITY))


def prep_define_file(dft_settings, charge):

    basisset = dft_settings["turbomole_basis"]
    functional = dft_settings["turbomole_functional"]

    try:
        from StringIO import StringIO as mStringIO
    except ImportError:
        from io import StringIO as mStringIO

    outfile = mStringIO()
    outfile.write("\n\n\n\n\na coord\n*\nno\n")
    outfile.write("\n\nb all %s\n\n\n" % basisset)
    if charge == +1 or charge == -1:
        outfile.write("*\neht\n\n%i\n\n\n\n\n\n\n\n\n" % int(charge))
        outfile.write("\n\n\n\n")
    else:
        outfile.write("*\neht\n\n\n\n\n\n\n")

    if functional != "HF":
        outfile.write("dft\non\nfunc %s\n\n\n" % functional)

    if dft_settings["turbomole_method"]=="ridft":
        outfile.write("ri\non\nm1500\n\n\n\n" )

    outfile.write("scf\niter\n1000\n\n\n")

    outfile.write("scf\ndsp\non\n\n\n")

    outfile.write("scf\nconv\n5\n\n\n")
    #if charge == +1 or charge == -1:
    #    outfile.write("damp\n8.0\n0.1\n0.1\n")

    outfile.write("\n\n\n\n*\n")
    returnstring = outfile.getvalue()
    outfile.close()
    return returnstring


def AddStatementToControl(controlfilename, statement):
    inf = open(controlfilename, 'r')
    lines = inf.readlines()
    inf.close()
    already_in=False
    outf = open(controlfilename, 'w')
    for line in lines:
        if statement.split()[0] in line:
            already_in=True
        if len(line.split()) > 0:
            if line.split()[0] == "$end" and not already_in:
                outf.write("%s\n" % (statement))
        outf.write(line)
    outf.close()


def RemoveStatementFromControl(controlfilename, statement):
    inf = open(controlfilename, 'r')
    lines = inf.readlines()
    inf.close()
    outf = open(controlfilename, 'w')
    writeOutput = True
    for line in lines:
        if len(line.split()) > 0:
            if line.split()[0] == statement.split()[0]:
                writeOutput = False
            else:
                writeOutput = True
        if writeOutput:
            outf.write(line)
    outf.close()



