// Demostracion exclusivamente GPU: reutiliza FiltroGPU y los kernels originales.
// No ejecuta una convolucion CPU, ni necesita referencias de otra implementacion.
#include "practica.hpp"
#include "filtros_gpu.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

int entero(const std::string& texto) {
    std::size_t usados=0;
    int n=std::stoi(texto,&usados);
    if (usados!=texto.size()) throw std::invalid_argument("Entero invalido.");
    return n;
}
int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> op;
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i];
            if(i+1>=argc || (key!="--input" && key!="--filter" && key!="--mask" &&
                key!="--threads" && key!="--reps" && key!="--output"))
                throw std::invalid_argument("Parametros invalidos; usar el ejecutor Python.");
            op[key]=argv[i+1];
        }
        for(const char* key:{"--input","--filter","--mask","--threads","--reps","--output"})
            if(!op.count(key)) throw std::invalid_argument(std::string("Falta ")+key);
        practica::Filtro tipo;
        if(op["--filter"]=="media") tipo=practica::Filtro::Media;
        else if(op["--filter"]=="nitidez") tipo=practica::Filtro::Nitidez;
        else if(op["--filter"]=="laplaciano") tipo=practica::Filtro::Laplaciano;
        else throw std::invalid_argument("Filtro invalido.");
        const int k=entero(op["--mask"]),hilos=entero(op["--threads"]),reps=entero(op["--reps"]);
        if(reps<1 || reps>1000 || (k!=3 && k!=9 && k!=21) ||
           (hilos!=128 && hilos!=256 && hilos!=512) ||
           (tipo!=practica::Filtro::Media && k!=3))
            throw std::invalid_argument("Mascara, hilos o repeticiones invalidos.");
        const auto imagen=practica::cargarPGM(op["--input"]);
        const std::string prefix=op["--output"];
        for(const char* suf:{"_raw.csv","_salida.f32","_dispositivo.txt"})
            if(std::filesystem::exists(prefix+suf)) throw std::runtime_error("El prefijo ya existe.");
        if(!std::filesystem::path(prefix).parent_path().empty())
            std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
        std::vector<float> salida(imagen.pixeles.size());
        practica::FiltroGPU gpu(imagen.ancho,imagen.alto,k,hilos,tipo);
        std::ofstream meta(prefix+"_dispositivo.txt");
        meta<<gpu.descripcion()<<"\nValidacion CPU: no realizada; demostracion GPU independiente.\n";
        if(!meta) throw std::runtime_error("Error al guardar metadatos.");
        std::cout<<gpu.descripcion()<<"Calentamiento GPU (1 ejecucion)\n";
        gpu.ejecutar(imagen.pixeles,salida);
        std::ofstream csv(prefix+"_raw.csv");
        if(!csv) throw std::runtime_error("No se pudo abrir CSV.");
        csv<<std::setprecision(12);
        csv<<"implementacion,filtro,ancho,alto,mascara,hilos,bloques,repeticion,h2d_ms,kernel_ms,d2h_ms,total_ms,unidad,estado,validacion_cpu\n";
        for(int r=1;r<=reps;++r) {
            auto t=gpu.ejecutar(imagen.pixeles,salida);
            for(float v:salida) if(!std::isfinite(v)) throw std::runtime_error("Resultado GPU no finito.");
            csv<<"CUDA,"<<op["--filter"]<<','<<imagen.ancho<<','<<imagen.alto<<','<<k<<','
               <<hilos<<','<<gpu.bloques()<<','<<r<<','<<t.h2d_ms<<','<<t.kernel_ms<<','
               <<t.d2h_ms<<','<<t.total_ms<<",ms,completada,no_realizada\n";
            csv.flush();
            if(!csv) throw std::runtime_error("Error al guardar CSV.");
            std::cout<<"CUDA C "<<r<<'/'<<reps<<": kernel="<<t.kernel_ms<<" ms, total="<<t.total_ms<<" ms\n";
        }
        std::ofstream raw(prefix+"_salida.f32",std::ios::binary);
        raw.write(reinterpret_cast<const char*>(salida.data()),
                  static_cast<std::streamsize>(salida.size()*sizeof(float)));
        if(!raw) throw std::runtime_error("Error al guardar salida float32.");
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;
    }
}
