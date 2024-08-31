/*
    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved
        Mingtian Zhao, Alexander D. MacKerell Jr.
    E-mail:
        zhaomt@outerbanks.umaryland.edu
        alex@outerbanks.umaryland.edu
*/


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "gcmc.h"

void print_atom(const Atom &atom) {
    std::cout << "  Position: (" << atom.position[0] << ", "
              << atom.position[1] << ", " << atom.position[2] << ")\t";
    std::cout << "  Charge: " << atom.charge << '\t';
    std::cout << "  Type: " << atom.type << '\n';
}

void print_atom_array(const AtomArray &atom_array) {
    std::string nameStr(atom_array.name, 4);
    std::cout << "Name: " << nameStr.substr(0, 4) << '\n';
    std::cout << "Muex: " << atom_array.muex << '\n';
    std::cout << "Conc: " << atom_array.conc << '\n';
    std::cout << "ConfBias: " << atom_array.confBias << '\n';
    std::cout << "McTime: " << atom_array.mcTime << '\n';

    std::cout << "StartRes: " << atom_array.startRes << '\n';

    std::cout << "TotalNum: " << atom_array.totalNum << '\n';
    std::cout << "MaxNum: " << atom_array.maxNum << '\n';

    std::cout << "Num_atoms: " << atom_array.num_atoms << '\n';
    for (int i = 0; i < atom_array.num_atoms; ++i) {
        std::cout << "Atom " << (i + 1) << ":\n";
        print_atom(atom_array.atoms[i]);
    }
}

void print_fragmentInfo(const AtomArray *fragmentInfo, int fragTypeNum) {

    std::cout << "\n\nPrinting FragmentInfo:\n\n";
    for (int i = 0; i < fragTypeNum; ++i) {
        std::cout << "AtomArray " << (i + 1) << ":\n";
        print_atom_array(fragmentInfo[i]);
        std::cout << '\n';
    }
}

void print_info_struct(const InfoStruct *info) {

    std::cout << "\n\nPrinting InfoStruct:\n\n";

    std::cout << "Mcsteps: " << info->mcsteps << '\n';
    std::cout << "Cutoff: " << info->cutoff << '\n';
    std::cout << "Grid_dx: " << info->grid_dx << '\n';
    std::cout << "Startxyz: (" << info->startxyz[0] << ", "
              << info->startxyz[1] << ", " << info->startxyz[2] << ")\n";
    std::cout << "Cryst: (" << info->cryst[0] << ", "
              << info->cryst[1] << ", " << info->cryst[2] << ")\n";

    std::cout << "CavityFactor: " << info->cavityFactor << '\n';
    std::cout << "FragTypeNum: " << info->fragTypeNum << '\n';
    std::cout << "TotalGridNum: " << info->totalGridNum << '\n';
    std::cout << "TotalResNum: " << info->totalResNum << '\n';
    std::cout << "TotalAtomNum: " << info->totalAtomNum << '\n';

}

void print_atoms(const AtomArray *fragmentInfo, int fragTypeNum, const residue *residueInfo, const Atom *atomInfo) {

    std::cout << "\n\nPrinting Atoms:\n\n";

    int startRes = fragmentInfo[0].startRes;
    int atomStart = residueInfo[startRes].atomStart;
    std::cout << "\nFixed Atoms " << atomStart << ":\n";

    for (int i = 0; i < atomStart; ++i) {
        std::cout << "Atom " << (i + 1) << ":\n";
        print_atom(atomInfo[i]);
    }
    std::cout << '\n';


    std::cout << "\nMoving Atoms:\n";
    for (int i = 0; i < fragTypeNum; ++i) {
        std::cout << "AtomArray " << (i + 1) << ":\n";
        for (int j = fragmentInfo[i].startRes; j < fragmentInfo[i].startRes + fragmentInfo[i].totalNum; ++j) {
            std::cout << "Residue " << (j + 1) << ":\n";
            for (int k = residueInfo[j].atomStart; k < residueInfo[j].atomStart + residueInfo[j].atomNum; ++k) {
                std::cout << "Atom " << (k + 1) << ":\n";
                print_atom(atomInfo[k]);
            }
        }
        std::cout << '\n';
    }
}


void print_grid(const float *grid, int totalGridNum) {
    std::cout << "\n\nPrinting Grid:\n\n";
    for (int i = 0; i < totalGridNum ; ++i) {
        std::cout << "Grid " << (i + 1) << ": " << grid[i * 3] << ", " << grid[i * 3 + 1] << ", " << grid[i * 3 + 2] << '\n';
    }
}

void print_ff(const float *ff, int ffXNum, int ffYNum) {
    std::cout << "\n\nPrinting FF:\n\n";
    for (int i = 0; i < ffXNum; ++i) {
        for (int j = 0; j < ffYNum; ++j) {
            std::cout << "FF " << (i + 1) << ", " << (j + 1) << ": " << " sigma " << ff[(i * ffYNum + j)*2] << ", " << " epsilon " << ff[(i * ffYNum + j)*2 + 1] << '\n';
        }
    }
}

void print_moveArray(const int *moveArray, int mcsteps, const AtomArray *fragmentInfo) {
    std::cout << "\n\nPrinting MoveArray:\n\n";
    for (int i = 0; i < mcsteps; ++i) {
        std::cout << "Move " << (i + 1) << ":\t";
        int fragType = moveArray[i] / 4;
        int moveType = moveArray[i] % 4;

        std::cout << "FragType: " ;

        // std::cout << fragType << '\t';
        std::string nameStr(fragmentInfo[fragType].name, 4);
        std::cout << nameStr.substr(0, 4) << '\t';

        std::cout << "MoveType: " << moveType << '\n';
    }
}

void print_all_info(const InfoStruct *info, const AtomArray *fragmentInfo, const residue *residueInfo, const Atom *atomInfo, const float *grid, const float *ff, const int *moveArray) {
    
    print_info_struct(info);
    print_fragmentInfo(fragmentInfo, info->fragTypeNum);
    print_atoms(fragmentInfo, info->fragTypeNum, residueInfo, atomInfo);
    print_grid(grid, info->totalGridNum);
    print_ff(ff, info->ffXNum, info->ffYNum);
    print_moveArray(moveArray, info->mcsteps, fragmentInfo);

    std::flush(std::cout);
    
}

extern "C" void runGCMC_cuda(const InfoStruct *info, AtomArray *fragmentInfo, residue *residueInfo, Atom *atomInfo, const float *grid, const float *ff, const int *moveArray);


static PyObject *runGCMC(PyObject *self, PyObject *args) {
    PyObject *py_info;
    PyObject *py_fragmentInfo;
    PyObject *py_residueInfo;
    PyObject *py_atomInfo;
    PyObject *py_grid;
    PyObject *py_ff;
    PyObject *py_moveArray;

    
    if (!PyArg_ParseTuple(args, "OOOOOOO", &py_info, &py_fragmentInfo, &py_residueInfo, &py_atomInfo, &py_grid, &py_ff, &py_moveArray)) {
        return NULL;
    }

    InfoStruct *info = (InfoStruct *)PyArray_DATA((PyArrayObject *)py_info);
    AtomArray *fragmentInfo = (AtomArray *)PyArray_DATA((PyArrayObject *)py_fragmentInfo);
    residue *residueInfo = (residue *)PyArray_DATA((PyArrayObject *)py_residueInfo);
    Atom *atomInfo = (Atom *)PyArray_DATA((PyArrayObject *)py_atomInfo);
    float *grid = (float *)PyArray_DATA((PyArrayObject *)py_grid);
    float *ff = (float *)PyArray_DATA((PyArrayObject *)py_ff);
    int *moveArray = (int *)PyArray_DATA((PyArrayObject *)py_moveArray);

    if (info->showInfo > 0)
        print_all_info(info, fragmentInfo, residueInfo, atomInfo, grid, ff, moveArray);

    
    runGCMC_cuda(info, fragmentInfo, residueInfo, atomInfo, grid, ff, moveArray);

    // std::cout << "Finished GCMC in GPU\n\n";

    if (info->showInfo > 0)
        print_all_info(info, fragmentInfo, residueInfo, atomInfo, grid, ff, moveArray);

    Py_RETURN_NONE;
}


static PyMethodDef methods[] = {
    {"runGCMC", runGCMC, METH_VARARGS, "Run GCMC with CUDA"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC PyInit_gpu(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "gpu",          /* m_name */
        "Module doc",   /* m_doc */
        -1,             /* m_size */
        methods,        /* m_methods */
        NULL,           /* m_reload */
        NULL,           /* m_traverse */
        NULL,           /* m_clear */
        NULL,           /* m_free */
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    import_array();  // Required for NumPy
    return m;
}