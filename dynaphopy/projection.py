import numpy as np
from dynaphopy.power_spectrum import _progress_bar
import time


def project_onto_wave_vector(trajectory, q_vector, project_on_atom=-1):

    start_time = time.time()
    number_of_primitive_atoms = trajectory.structure.get_number_of_primitive_atoms()
    velocity = trajectory.get_velocity_mass_average()
    #    velocity = trajectory.velocity   # (use the velocity without mass average, just for testing)

    number_of_atoms = velocity.shape[1]
    number_of_dimensions = velocity.shape[2]
    supercell = trajectory.get_supercell_matrix()

    coordinates = trajectory.structure.get_positions(supercell)
    atom_type = trajectory.structure.get_atom_type_index(supercell=supercell)

    velocity_projected = np.zeros(
        (velocity.shape[0], number_of_primitive_atoms, number_of_dimensions),
        dtype=complex,
    )

    if q_vector.shape[0] != coordinates.shape[1]:
        print("Warning!! Q-vector and coordinates dimension do not match")
        exit()

    # Projection into wave vector
    for i in range(number_of_atoms):
        # Projection on atom
        if project_on_atom > -1:
            if atom_type[i] != project_on_atom:
                continue

        for k in range(number_of_dimensions):
            velocity_projected[:, atom_type[i], k] += velocity[:, i, k] * np.exp(
                -1j * np.dot(q_vector, coordinates[i, :])
            )

    # Normalize velocities (method 1)
    #  for i in range(velocity_projected.shape[1]):
    #      velocity_projected[:,i,:] /= atom_type.count(i)

    # Normalize velocities (method 2)
    number_of_primitive_cells = number_of_atoms / number_of_primitive_atoms
    velocity_projected /= np.sqrt(number_of_primitive_cells)
    print("Projected trajectory on wave vector in %.2f s" % (time.time() - start_time))
    return velocity_projected


def compute_norm_vels(
    qvT,
    number_of_primitive_atoms,
    project_on_atom,
    atom_type,
    atom_vels,
    coordinates,
    sqmasses,
):
    # SW: function for parallelization
    norm_vels = np.zeros((number_of_primitive_atoms, 3), dtype=complex)
    for pid in range(number_of_primitive_atoms):
        # Projection on atom
        if project_on_atom > -1:
            if pid != project_on_atom:
                continue
        atom_list = pid == atom_type
        norm_vels[pid, :] = (
            np.dot(
                atom_vels[atom_list, :].T,
                np.exp(-1j * np.dot(coordinates[atom_list, :], qvT)),
            )
            * sqmasses[pid]
        )
    return norm_vels


def project_onto_wave_vector_lammps_vel(
    fname,
    trajectory,
    q_vector,
    project_on_atom=-1,
    first_vel=1,
    interval=1,
    start_vel=1,
    end_vel=None,
    silent=False,
):
    """
    added by Sandro Wieser
    memory light variant where the velocities are read on the fly instead of relying on a saved trajectory in the memory
    the velocities read are those specific to lammps
    first_vel ... first velocity column index in lammps file
    interval ... interpret only every nth time step
    start_vel ... first velocity to read
    end_vel ... maximum number of time steps to read
    """

    start_time = time.time()
    # Note: trajectory is a Dynamics object
    number_of_primitive_atoms = trajectory.structure.get_number_of_primitive_atoms()
    # if this does not work, need to compute from the supercell
    sqmasses = np.sqrt(np.array(trajectory.structure.get_masses()))
    supercell = (
        trajectory.get_supercell_matrix()
    )  # it is not a matrix, the case of it being one is still considered
    if len(supercell.ravel()) == 3:
        sc_multiplicity = np.prod(supercell)
    else:
        sc_multiplicity = int(round(np.linalg.det(supercell)))

    number_of_atoms = number_of_primitive_atoms * sc_multiplicity
    q_vector_T = []
    velocity_projected = []
    for qv in q_vector:
        q_vector_T.append(qv.T)
        velocity_projected.append([])

    # fetches masses of the primitive uc

    coordinates = trajectory.structure.get_positions(supercell)
    atom_type = np.array(trajectory.structure.get_atom_type_index(supercell=supercell))

    if np.shape(q_vector[0])[0] != coordinates.shape[1]:
        print("ERROR! Q-vector and coordinates dimension do not match")
        exit()

    if (not silent) & (end_vel is not None):
        num_to_read = end_vel - start_vel + 1
        print(
            "Reading %d time steps with %d atoms each..."
            % (num_to_read, number_of_atoms)
        )
        _progress_bar(0, "Reading trajectory")

    first_vel -= 1  # start counting at 0
    # Here, we read the velocities and evaluate them on the fly
    computation_time = 0
    import multiprocessing

    with open(fname) as fp:
        atom_vels = np.zeros((number_of_atoms, 3))
        aid = None
        readmode = False
        tsteps_passed = 0

        with multiprocessing.Pool() as pool:
            for line in fp:
                elements = line.split()
                if len(elements) >= 3:  # individual atoms
                    if (elements[0] != "ITEM:") & readmode:
                        atom_vels[aid] = np.array(
                            list(map(float, elements[first_vel : first_vel + 3]))
                        )
                        aid += 1
                        if aid == number_of_atoms:
                            readmode = False
                            # do the processing
                            if start_vel <= (tsteps_passed + 1):
                                ctime1 = time.time()
                                # Projection into wave vector
                                # parallel version

                                for qid, nv in enumerate(
                                    pool.starmap(
                                        compute_norm_vels,
                                        [
                                            (
                                                qvT,
                                                number_of_primitive_atoms,
                                                project_on_atom,
                                                atom_type,
                                                atom_vels,
                                                coordinates,
                                                sqmasses,
                                            )
                                            for qvT in q_vector_T
                                        ],
                                    )
                                ):
                                    velocity_projected[qid].append(nv)

                                # serial version
                                #                                for qid, qvT in enumerate(q_vector_T):
                                #                                    norm_vels = np.zeros((number_of_primitive_atoms,3),dtype=complex)
                                #                                    for pid in range(number_of_primitive_atoms):
                                #                                        # Projection on atom
                                #                                        if project_on_atom > -1:
                                #                                            if pid != project_on_atom:
                                #                                                continue
                                #                                        atom_list = pid == atom_type
                                #        #                                print(np.shape(mult_qvec),np.shape(coordinates[atom_list,:]),np.shape(atom_vels[atom_list,:]),
                                #        #                                      np.shape(np.dot(coordinates[atom_list,:],q_vector.T)))
                                #                                        norm_vels[pid, :] = np.dot(atom_vels[atom_list,:].T,
                                #                                                                   np.exp(-1j*np.dot(coordinates[atom_list,:],qvT))) * \
                                #                                                                   sqmasses[pid]
                                #
                                #                                    velocity_projected[qid].append(norm_vels)
                                computation_time += time.time() - ctime1

                                if (not silent) & (end_vel is not None):
                                    _progress_bar(
                                        float(tsteps_passed + 1 - start_vel)
                                        / num_to_read,
                                        "Reading trajectory",
                                    )
                            tsteps_passed += 1
                            if end_vel is not None:
                                if tsteps_passed >= end_vel:
                                    break
                    elif (elements[0] == "ITEM:") & (elements[1] == "ATOMS"):
                        readmode = True
                        aid = 0

    #    velocity = trajectory.velocity   # (use the velocity without mass average, just for testing)

    # Normalize velocities (method 1)
    #  for i in range(velocity_projected.shape[1]):
    #      velocity_projected[:,i,:] /= atom_type.count(i)

    # Normalize velocities (method 2)

    velocity_projected = np.array(velocity_projected)
    number_of_primitive_cells = number_of_atoms / number_of_primitive_atoms
    velocity_projected /= np.sqrt(number_of_primitive_cells)
    print(
        "Read and projected trajectory on wave vector in %.2f s with %.2f s computation time"
        % (time.time() - start_time, computation_time)
    )
    return velocity_projected


def project_onto_phonon(vc, eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    # Projection in phonon coordinate
    velocity_projected = np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)
    for k in range(number_of_frequencies):
        for i in range(number_of_cell_atoms):
            velocity_projected[:, k] += np.dot(
                vc[:, i, :], eigenvectors[k, i, :].conj()
            )

    return velocity_projected


# Just for testing (slower implementation) [but equivalent]
def project_onto_phonon2(vc, eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    # Projection in phonon coordinate
    velocity_projected = np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)

    for i in range(vc.shape[0]):
        for k in range(number_of_frequencies):
            velocity_projected[i, k] = np.trace(
                np.dot(vc[i, :, :], eigenvectors[k, :, :].T.conj())
            )
    #            velocity_projected[i,k] = np.sum(np.linalg.eigvals(np.dot(vc[i,:,:],eigenvectors[k,:,:].T.conj())))
    return velocity_projected
