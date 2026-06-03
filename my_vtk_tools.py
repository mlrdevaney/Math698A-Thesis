########################
# 06/01/2020 A. Alekseenko
# This module contains functions and
# subroutines that create .vtk files
########################

#########################
# my_get_svdsols_file_names
# This subroutine returns a list
# of filenames that are located in
# directory given by path. Files will be
# located in subdirectories and listed with
# path, starting with the provided path.
#
###########################
def my_get_svdsols_file_names (path):
    import os

    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'svd_SolET_' in file:
                files.append(os.path.join(r, file))

    return files


##################################################
# writeVTKsolcollpointvals (solarry_0, collarry_0, nodes_u, nodes_v, nodes_w, timestamp, path)
#
# This subroutine will prepare two VTK files. One will contain containing point mesh, values of velocity
# distribution function and absolute values of the velocity distribution function. and collision.
#
# The other one will contain point mesh and values of the collision operator and absolute values of the collision
# operator.
#
# the names are plotsol_TIMESTAMP.vtp and plotcoll_TIMESTAMP.vtp
#
# input:
# solarray_0[1,.] numpy array containing a copy of the solution array. First dimension is 1 record long. Second should
#             correspond to the number of velocity nodes
# collarray_0[1,.] numpy array containing a copy of collision operator. First dimension is 1 record long. Second should
#              correspond to the number of velocity nodes
# nodes_u, nodes_v, nodes_w -- arrays of containing u,v,w components of the meshpoints in 3D velocity space.
# timestamp --- real number indicating the value of time for the solution save -- is sueful for making movies of solutions
# path  --- path to folder to where the place the file resulting file.
##################################################
def writeVTKsolcollpointvals (solarry_0, collarry_0, nodes_u, nodes_v, nodes_w, timestamp, path):

 import numpy as np

 n1 = nodes_u.shape[1]  # total number of nodal points
 vtk_file_path = path + '\\plotsol_{0:8.8f}.vtp'.format(timestamp)
 save_file = open(vtk_file_path, 'w')
 ## Now the file is opened, we will write stuff into it.
 save_file.write('<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\"> \n')
 save_file.write('  <PolyData>\n')  # ints ' + 'I{0:02d}J{1:02d}K{2:02d}\n'.format(ii1,ii2,ii4))
 save_file.write('    <Piece NumberOfPoints=\"{0}\" NumberOfVerts=\"0\" '.format(n1) +
                 'NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n')
 save_file.write('      <Points>\n')
 save_file.write('        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n')
 data_line = ''
 for ii3 in range(n1):
        data_line = data_line + '{0:8.5f} '.format(nodes_u[0, ii3])
        data_line = data_line + '{0:8.5f} '.format(nodes_v[0, ii3])
        data_line = data_line + '{0:8.5f} '.format(nodes_w[0, ii3])
        data_line = data_line + '  '
        if (ii3 + 1) % 10 == 0 and ii3 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 save_file.write('      </Points>\n')
 save_file.write('      <PointData Scalars=\"solution\">\n')
 print('Solution is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                                                                                       np.amax(solarry_0[0, :]),
                                                                                       np.amin(solarry_0[0, :])))
 save_file.write('        <DataArray type=\"Float32\" Name=\"solution\" format=\"ascii\">\n')
 data_line = ''
 for ii4 in range(0, n1):
        data_line = data_line + '{0:8.5f} '.format(
            solarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
        if (ii4 + 1) % 20 == 0 and ii4 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 print('Solution is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
                                                                                      np.amax(
                                                                                          np.absolute(solarry_0[0, :])),
                                                                                      np.amin(np.absolute(
                                                                                          solarry_0[0, :]))))
 save_file.write('        <DataArray type=\"Float32\" Name=\"solABS\" format=\"ascii\">\n')
 data_line = ''
 for ii4 in range(0, n1):
        if np.absolute(solarry_0[0, ii4]) > 0.0001:
            data_line = data_line + '{0:8.5f} '.format(np.absolute(solarry_0[0, ii4]))
        else:
            data_line = data_line + '{0:8.5f} '.format(0.0001)
        if (ii4 + 1) % 20 == 0 and ii4 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 save_file.write('      </PointData>\n')
 #############################
 save_file.write('    </Piece>\n')
 save_file.write('  </PolyData>\n')
 save_file.write('</VTKFile>\n')
 #### all done, the points are written. Close the file
 save_file.close()
 #####################################################################################################
 ###
 ### NEXT WE WRITE COLLISION SAVE IN A SEPARATE FILE.
 ###
 #####################################################################################################
 #### First, we need to get a name of the file.
 vtk_file_path = path + '\\plotcoll{0:8.8f}.vtp'.format(timestamp)
 save_file = open(vtk_file_path, 'w')
 ## Now the file is opened, we will write stuff into it.
 save_file.write('<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\"> \n')
 save_file.write('  <PolyData>\n')  # ints ' + 'I{0:02d}J{1:02d}K{2:02d}\n'.format(ii1,ii2,ii4))
 save_file.write('    <Piece NumberOfPoints=\"{0}\" NumberOfVerts=\"0\" '.format(n1) +
                    'NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n')
 save_file.write('      <Points>\n')
 save_file.write('        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n')
 data_line = ''
 for ii3 in range(n1):
        data_line = data_line + '{0:8.5f} '.format(nodes_u[0, ii3])
        data_line = data_line + '{0:8.5f} '.format(nodes_v[0, ii3])
        data_line = data_line + '{0:8.5f} '.format(nodes_w[0, ii3])
        data_line = data_line + '  '
        if (ii3 + 1) % 10 == 0 and ii3 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 save_file.write('      </Points>\n')
 save_file.write('      <PointData Scalars=\"solution\">\n')
 print('Solution is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                                np.amax(collarry_0[0, :]),np.amin(collarry_0[0, :])))
 save_file.write('        <DataArray type=\"Float32\" Name=\"colloperat\" format=\"ascii\">\n')
 data_line = ''
 for ii4 in range(0, n1):
        data_line = data_line + '{0:8.5f} '.format(
            collarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
        if (ii4 + 1) % 20 == 0 and ii4 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 print('Solution is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
            np.amax(np.absolute(collarry_0[0, :])), np.amin(np.absolute(collarry_0[0, :]))))
 save_file.write('        <DataArray type=\"Float32\" Name=\"collABS\" format=\"ascii\">\n')
 data_line = ''
 for ii4 in range(0, n1):
        if np.absolute(collarry_0[0, ii4]) > 0.00001:
            data_line = data_line + '{0:8.5f} '.format(np.absolute(collarry_0[0, ii4]))
        else:
            data_line = data_line + '{0:8.5f} '.format(0.0001)
        if (ii4 + 1) % 20 == 0 and ii4 > 0:
            save_file.write(data_line + '\n')
            data_line = ''
 save_file.write(data_line + '\n')
 save_file.write('        </DataArray>\n')
 save_file.write('      </PointData>\n')
 #############################
 save_file.write('    </Piece>\n')
 save_file.write('  </PolyData>\n')
 save_file.write('</VTKFile>\n')
 #### all done, the points are written. Close the file
 save_file.close()
 ############# ALL DONE


##################################################
# writeVTKsolcollcellvalsRLG(solarray_0, collarry_0,grids_u, grids_v, grids_w, timestamp, path)
#
# This subroutine will prepare a VTK file with cell values of velocity
# distribution function and collision operator and absolute values of the velocity distribution function and collision.
#
#
# TYPE OF THE VTK FILE: RectilinearGrid -- hence the "RLG" suffix
#
#
# the names are plotsol_TIMESTAMP.vtp and plotcoll_TIMESTAMP.vtp
#
# input:
# solarray_0[1,.] numpy array containing a copy of the solution array. First dimension is 1 record long. Second should
#             correspond to the number of velocity nodes
# collarray_0[1,.] numpy array containing a copy of collision operator. First dimension is 1 record long. Second should
#              correspond to the number of velocity nodes
# grids_u, grids_v, grids_w -- arrays of containing u,v,w grids that make the rectilinear grid in 3D velocity space.
# timestamp --- real number indicating the value of time for the solution save -- is sueful for making movies of solutions
# path  --- path to folder to where the place the file resulting file.
##################################################
def writeVTKsolcollcellvalsRLG(solarry_0, collarry_0, grids_u, grids_v, grids_w, timestamp, path):
     import numpy as np
     ## Get some dimentions that we will need for our work
     ngru = grids_u.shape[1]  # number of grid points in u variable
     ngrv = grids_v.shape[1]  # number of grid points in v variable
     ngrw = grids_w.shape[1]  # number of grid points in w variable
     ##
     n1 = solarry_0.shape[1]  # total number of nodal points
     ##
     vtk_file_path = path + '\\plotsolcollcellsRLG_{0:8.8f}.vtr'.format(timestamp)
     save_file = open(vtk_file_path, 'w')
     ## Now the file is opened, we will write stuff into it.
     save_file.write('<?xml version = "1.0"?>\n')
     save_file.write('<VTKFile type=\"RectilinearGrid\" version=\"0.1\" ' +
                                'byte_order=\"LittleEndian\" > \n')
     save_file.write('  <RectilinearGrid WholeExtent = \"{0} {1} {2} {3} {4} {5}\">\n'.format(0,ngru-1,0,ngrv-1,0,ngrw-1))
     save_file.write('    <FieldData>\n')
     save_file.write('      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples = \"1\" format=\"ascii\">\n')
     save_file.write('      {0:8.8f}\n'.format(timestamp))
     save_file.write('      </DataArray>\n')
     save_file.write('    </FieldData>\n')
     save_file.write('    <Piece Extent = \"{0} {1} {2} {3} {4} {5}\">\n'.format(0,ngru-1,0,ngrv-1,0,ngrw-1))
     save_file.write('      <CellData>\n')
     ########################## Debug info
     print('Solution is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                                np.amax(solarry_0[0, :]),np.amin(solarry_0[0, :])))
     ########################## end debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solution\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         data_line = data_line + '{0:8.5f} '.format(
             solarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('Absolute value of solution is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
                    np.amax(np.absolute(solarry_0[0, :])),np.amin(np.absolute(solarry_0[0, :]))))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solABS\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         if np.absolute(solarry_0[0, ii4]) > 0.0001:
             data_line = data_line + '{0:8.5f} '.format(np.absolute(solarry_0[0, ii4]))
         else:
             data_line = data_line + '{0:8.5f} '.format(0.0001)
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('Collision operator is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                   np.amax(collarry_0[0, :]),np.amin(collarry_0[0, :])))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"colloperat\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         data_line = data_line + '{0:8.5f} '.format(
             collarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('Absolute values of collision operator is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
                               np.amax(np.absolute(collarry_0[0, :])),np.amin(np.absolute(collarry_0[0, :]))))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"collABS\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         if np.absolute(collarry_0[0, ii4]) > 0.00001:
             data_line = data_line + '{0:8.5f} '.format(np.absolute(collarry_0[0, ii4]))
         else:
             data_line = data_line + '{0:8.5f} '.format(0.0001)
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     save_file.write('      </CellData>\n')
     #############################
     save_file.write('      <Coordinates>\n')
     ########################## Debug info
     print('grip points in variable u are about to be recorded max U_L={0} min U_R={1}'.format(grids_u[0,0],
                grids_u[0,ngru-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_u\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngru):
         data_line = data_line + '{0:8.5f} '.format(grids_u[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('grip points in variable v are about to be recorded max V_L={0} min V_R={1}'.format(grids_v[0,0],
                grids_v[0,ngrv-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_v\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngrv):
         data_line = data_line + '{0:8.5f} '.format(grids_v[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('grip points in variable w are about to be recorded max W_L={0} min W_R={1}'.format(grids_w[0,0],
                grids_w[0,ngrw-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_w\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngrw):
         data_line = data_line + '{0:8.5f} '.format(grids_w[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ############################
     save_file.write('      </Coordinates>\n')
     save_file.write('    </Piece>\n')
     save_file.write('  </RectilinearGrid>\n')
     save_file.write('</VTKFile>\n')
     #### all done, the points are written. Close the file
     save_file.close()
     ############# ALL DONE

##################################################
# writeVTKsolcellvalsRLG(solarray_0, grids_u, grids_v, grids_w, timestamp, path, vtkfname)
#
# This subroutine will prepare a VTK file with cell values of velocity
# distribution function and collision operator and absolute values of the velocity distribution function and collision.
#
# This subroutine only works for s=1 --- one point per cell hense can use cell data.
#
# TYPE OF THE VTK FILE: RectilinearGrid -- hence the "RLG" suffix
#
#
# the names are plotsol_TIMESTAMP.vtp and plotcoll_TIMESTAMP.vtp
#
# input:
# solarray_0[1,.] numpy array containing a copy of the solution array. First dimension is 1 record long. Second should
#             correspond to the number of velocity nodes
# collarray_0[1,.] numpy array containing a copy of collision operator. First dimension is 1 record long. Second should
#              correspond to the number of velocity nodes
# grids_u, grids_v, grids_w -- arrays of containing u,v,w grids that make the rectilinear grid in 3D velocity space.
# timestamp --- real number indicating the value of time for the solution save -- is sueful for making movies of solutions
# path  --- path to folder to where the place the file resulting file.
##################################################
def writeVTKsolcellvalsRLG(solarry_0, grids_u, grids_v, grids_w, timestamp, path, vtkfname):

     import numpy as np
     ## Get some dimentions that we will need for our work
     ngru = grids_u.shape[1]  # number of grid points in u variable
     ngrv = grids_v.shape[1]  # number of grid points in v variable
     ngrw = grids_w.shape[1]  # number of grid points in w variable
     ##
     n1 = solarry_0.shape[1]  # total number of nodal points
     ##
     vtk_file_path = path + '\\' + vtkfname +'_{0:04d}.vtr'.format(int(timestamp))
     save_file = open(vtk_file_path, 'w')
     ## Now the file is opened, we will write stuff into it.
     save_file.write('<?xml version = "1.0"?>\n')
     save_file.write('<VTKFile type=\"RectilinearGrid\" version=\"0.1\" ' +
                                'byte_order=\"LittleEndian\" > \n')
     save_file.write('  <RectilinearGrid WholeExtent = \"{0} {1} {2} {3} {4} {5}\">\n'.format(0,ngru-1,0,ngrv-1,0,ngrw-1))
     save_file.write('    <FieldData>\n')
     save_file.write('      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples = \"1\" format=\"ascii\">\n')
     save_file.write('      {0:8.8f}\n'.format(timestamp))
     save_file.write('      </DataArray>\n')
     save_file.write('    </FieldData>\n')
     save_file.write('    <Piece Extent = \"{0} {1} {2} {3} {4} {5}\">\n'.format(0,ngru-1,0,ngrv-1,0,ngrw-1))
     save_file.write('      <CellData>\n')
     ########################## Debug info
     print('Solution is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                                np.amax(solarry_0[0, :]),np.amin(solarry_0[0, :])))
     ########################## end debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solution\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         data_line = data_line + '{0:8.5f} '.format(
             solarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('Absolute value of solution is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
                    np.amax(np.absolute(solarry_0[0, :])),np.amin(np.absolute(solarry_0[0, :]))))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solABS\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         if np.absolute(solarry_0[0, ii4]) > 0.0001:
             data_line = data_line + '{0:8.5f} '.format(np.absolute(solarry_0[0, ii4]))
         else:
             data_line = data_line + '{0:8.5f} '.format(0.0001)
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     save_file.write('      </CellData>\n')
     #############################
     save_file.write('      <Coordinates>\n')
     ########################## Debug info
     print('grip points in variable u are about to be recorded max U_L={0} min U_R={1}'.format(grids_u[0,0],
                grids_u[0,ngru-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_u\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngru):
         data_line = data_line + '{0:8.5f} '.format(grids_u[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('grip points in variable v are about to be recorded max V_L={0} min V_R={1}'.format(grids_v[0,0],
                grids_v[0,ngrv-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_v\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngrv):
         data_line = data_line + '{0:8.5f} '.format(grids_v[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('grip points in variable w are about to be recorded max W_L={0} min W_R={1}'.format(grids_w[0,0],
                grids_w[0,ngrw-1]))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"grid_w\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, ngrw):
         data_line = data_line + '{0:8.5f} '.format(grids_w[0, ii4])
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ############################
     save_file.write('      </Coordinates>\n')
     save_file.write('    </Piece>\n')
     save_file.write('  </RectilinearGrid>\n')
     save_file.write('</VTKFile>\n')
     #### all done, the points are written. Close the file
     save_file.close()
     ############# ALL DONE


##################################################
# writeVTKsolcellvalsPolyData(solarray_0, grids_u, grids_v, grids_w, nodes_u, nodes_v, nodes_w, timestamp, path, vtkfname)
#
# This subroutine will prepare a VTK file with point values of velocity
# distribution function and absolute values of the velocity distribution function.
#
# TYPE OF THE VTK FILE: PolyData -- hence the "PolyData" suffix
#
#
# the names are plotsol_TIMESTAMP.pvtp
#
# input:
# solarray_0[1,.] numpy array containing a copy of the solution array. First dimension is 1 record long. Second should
#             correspond to the number of velocity nodes
#              correspond to the number of velocity nodes
# grids_u, grids_v, grids_w -- arrays of containing u,v,w grids that make the rectilinear grid in 3D velocity space.
# nodes_u, nodes_v, nodes_w -- arrays containing collocation points where solution is defined.
# timestamp --- real number indicating the value of time for the solution save -- is sueful for making movies of solutions
# path  --- path to folder to where the place the file resulting file.
##################################################
def writeVTKsolcellvalsPolyData(solarry_0, grids_u, grids_v, grids_w, nodes_u, nodes_v, nodes_w, timestamp, path, vtkfname):
     import numpy as np
     import os
     ## Get some dimentions that we will need for our work
     ngru = grids_u.shape[1]  # number of grid points in u variable
     ngrv = grids_v.shape[1]  # number of grid points in v variable
     ngrw = grids_w.shape[1]  # number of grid points in w variable
     ##
     nndsu = nodes_u.shape[1]  # number of nodal points in u variable
     nndsv = nodes_v.shape[1]  # number of nodal points in v variable
     nndsw = nodes_w.shape[1]  # number of nodal points in w variable

     n1 = solarry_0.shape[1]  # total number of nodal points
     ##
     vtk_file_path = os.path.join(path, vtkfname + '_{0:04d}.vtp'.format(int(timestamp)))  #CHANGED
     save_file = open(vtk_file_path, 'w')
     ## Now the file is opened, we will write stuff into it.
     save_file.write('<?xml version = "1.0"?>\n')
     save_file.write('<VTKFile type=\"PolyData\" version=\"0.1\" ' +
                                'byte_order=\"LittleEndian\" > \n')
     save_file.write('  <PolyData>\n')
     save_file.write('    <Piece NumberOfPoints=\"{0}\"  NumberOfVerts=\"0\" '.format(n1) +
                                'NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n')
     ####################### now we record points where the point data belong ####
     save_file.write('      <Points>\n')
     save_file.write('        <DataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">\n')
     ########################## Debug info
     print('collocation nodea are about to be recorded. Lengths of component arrays: len(nodes_u)={0} '.format(nndsu) +
           'len(nodes_v)={0}, len(nodes_w)={1}'.format(nndsv, nndsw))
     ########################## end debug info
     data_line = ''
     for ii4 in range(0, n1):
         data_line = data_line + '{0} '.format(nodes_u[0, ii4])
         data_line = data_line + '{0} '.format(nodes_v[0, ii4])
         data_line = data_line + '{0} '.format(nodes_w[0, ii4])
         if (ii4 + 1) % 10 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     save_file.write('      </Points>\n')
     ##########################################
     save_file.write('      <PointData>\n')
     ########################## Debug info ####
     print('Solution is about to be recorded time={0}. max val={1}, min val={2}'.format(timestamp,
                                np.amax(solarry_0[0, :]),np.amin(solarry_0[0, :])))
     ########################## end debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solution\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         data_line = data_line + '{0:8.5f} '.format(
             solarry_0[0, ii4])  # this may will ran outside of array bounds if array is inconsistent
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     ########################## Debug info
     print('Absolute value of solution is about to be recorded time={0}. max val={1} min val={2}'.format(timestamp,
                    np.amax(np.absolute(solarry_0[0, :])),np.amin(np.absolute(solarry_0[0, :]))))
     ########################## end Debug info
     save_file.write('        <DataArray type=\"Float32\" Name=\"solABS\" NumberOfComponents=\"1\" format=\"ascii\">\n')
     data_line = ''
     for ii4 in range(0, n1):
         if np.absolute(solarry_0[0, ii4]) > 0.0001:
             data_line = data_line + '{0:8.5f} '.format(np.absolute(solarry_0[0, ii4]))
         else:
             data_line = data_line + '{0:8.5f} '.format(0.0001)
         if (ii4 + 1) % 20 == 0 and ii4 > 0:
             save_file.write(data_line + '\n')
             data_line = ''
     save_file.write(data_line + '\n')
     save_file.write('        </DataArray>\n')
     save_file.write('      </PointData>\n')
     save_file.write('    </Piece>\n')
     save_file.write('  </PolyData>\n')
     save_file.write('</VTKFile>\n')
     #### all done, the points are written. Close the file
     save_file.close()
     ############# ALL DONE
