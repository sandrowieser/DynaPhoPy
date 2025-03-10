import numpy as np

import subprocess

# support old phonopy versions
'''
  Variation of phonopy_link modified by sw - the goal is to allow reading phana .bin files to obtain the eigenvectors
  phonopy is still used for many other features - not everything was adapted.
'''


def read_phana_dyn(fname):
    with open(fname) as fp:
        dynmat = []
        for line in fp:
            if "q" in line:
                qpoint = list(map(float,line.split("[")[1].split("]")[0].split()))
            else:
                elements = line.split()
                if len(elements) > 0:
                    dynmat.append(np.array(list(map(complex,elements[0:len(elements):2])))+
                                  1j*np.array(list(map(complex,elements[1:len(elements):2]))))
    return np.array(dynmat)

def obtain_phana_eigenvectors_and_frequencies(q_vector, test_orthonormal=False, print_data=True, fname="phana.bin", execname="phana_linux"):
    
    with open("in.dynmat", "w") as ofp:
        ofp.write(fname+"\n")
        ofp.write("1\n1\n3\n")
        ofp.write(" ".join(list(map(str,q_vector)))+"\n")
        ofp.write("dynmat.dat\n0\n")
        ofp.close()

    with open("in.dynmat") as fp:
        cmd = [execname]

        p = subprocess.Popen(cmd, stdin=fp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        strings, err = p.communicate()
    
    
    dmat = read_phana_dyn("dynmat.dat")
    conversion2THz = 15.633302419162298
    freqs, evecs = np.linalg.eigh(dmat)
    freqs = freqs.astype(complex)**0.5*conversion2THz
    freqs = (np.real(freqs) - np.imag(freqs)).astype(float)
    #natoms = int(len(freqs)/3)
    #e_new = np.zeros((len(evecs[0]),natoms,3),dtype='complex')
    #for eid in range(len(evecs)):
    #    e_new[eid,:] = evecs[:,eid].reshape((natoms,3))
    #evecs = e_new
    
    number_of_dimensions = 3
    number_of_primitive_atoms = int(round(len(freqs)/3))
    
    arranged_ev = np.array([[[evecs [j*number_of_dimensions+k, i]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_primitive_atoms)]
                                    for i in range(number_of_primitive_atoms*number_of_dimensions)])
    
    if print_data:
        print("Harmonic frequencies (THz):")
        print(freqs)
        
    return arranged_ev, freqs
