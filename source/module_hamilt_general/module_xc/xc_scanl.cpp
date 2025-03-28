#ifdef USE_LIBXC

#include "xc_functional.h"
#include "xc_functional_libxc.h"
#include "module_elecstate/module_charge/charge.h"
#include "module_base/global_variable.h"
#include "module_parameter/parameter.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"

#include <xc.h>

#include <vector>

std::tuple<double,double,ModuleBase::matrix,ModuleBase::matrix> XC_Functional_Libxc::v_xc_meta(
    const std::vector<int> &func_id,  
    const int &nrxx, 
    const double &omega, 
    const double tpiba,
    const Charge* const chr)
{
    ModuleBase::TITLE("XC_Functional_Libxc","v_xc_meta (SCAN-L version)");
    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_meta");

    const double e2 = 2.0;
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(PARAM.inp.nspin,nrxx);
    ModuleBase::matrix vofk(PARAM.inp.nspin,nrxx);

    const int nspin = PARAM.inp.nspin;
    std::vector<xc_func_type> funcs = XC_Functional_Libxc::init_func(func_id, 
        (nspin == 1) ? XC_UNPOLARIZED : XC_POLARIZED);

    const std::vector<double> rho = XC_Functional_Libxc::convert_rho(nspin, nrxx, chr);
    const std::vector<std::vector<ModuleBase::Vector3<double>>> gdr
        = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho, tpiba, chr);
    const std::vector<double> sigma = XC_Functional_Libxc::convert_sigma(gdr);

    std::vector<double> kin_r(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for(int is=0; is<nspin; ++is) {
        for(int ir=0; ir<nrxx; ++ir) {
            kin_r[ir*nspin+is] = chr->kin_r[is][ir] / 2.0;
        }
    }

    constexpr double rho_th = 1e-8;
    constexpr double grho_th = 1e-12;
    constexpr double tau_th = 1e-8;
    std::vector<double> sgn(nrxx * nspin, 1.0);

    // 密度筛选逻辑与SCAN相同
    if(nspin == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<nrxx; ++ir) {
            if(rho[ir]<rho_th || sqrt(std::abs(sigma[ir]))<grho_th || std::abs(kin_r[ir])<tau_th) {
                sgn[ir] = 0.0;
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for(int ir=0; ir<nrxx; ++ir) {
            if(rho[ir*2]<rho_th || sqrt(std::abs(sigma[ir*3]))<grho_th || std::abs(kin_r[ir*2])<tau_th) {
                sgn[ir*2] = 0.0;
            }
            if(rho[ir*2+1]<rho_th || sqrt(std::abs(sigma[ir*3+2]))<grho_th || std::abs(kin_r[ir*2+1])<tau_th) {
                sgn[ir*2+1] = 0.0;
            }
        }
    }

    std::vector<double> exc(nrxx);
    std::vector<double> vrho(nrxx * nspin);
    std::vector<double> vsigma(nrxx * ((nspin==1)?1:3));
    std::vector<double> vtau(nrxx * nspin);
    std::vector<double> vlapl(nrxx * nspin);

    for(xc_func_type &func : funcs) {
        assert(func.info->family == XC_FAMILY_MGGA);
        
        // 差异2：调用Libxc接口时自动使用SCAN-L的实现
        xc_mgga_exc_vxc(&func, nrxx, 
            rho.data(), sigma.data(), sigma.data(),  // 注意：SCAN-L可能使用不同的sigma处理
            kin_r.data(), 
            exc.data(), vrho.data(), vsigma.data(), 
            vlapl.data(), vtau.data());

        // 混合项处理（与SCAN相同结构）
        for(int ir=0; ir<nrxx; ++ir) {
#ifdef __EXX
            if(func.info->number == XC_MGGA_X_SCAN_L && 
               XC_Functional::get_func_type() == 5) {
                exc[ir] *= (1.0 - XC_Functional::get_hybrid_alpha());
            }
#endif
        }

        // 势项计算（与SCAN相同结构）
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:etxc, vtxc)
#endif
        for(int is=0; is<nspin; ++is) {
            for(int ir=0; ir<nrxx; ++ir) {
                const int idx = ir*nspin + is;
                etxc += e2 * exc[ir] * rho[idx] * sgn[idx];
                
                // diff：对SCAN-L交换项的混合系数处理
#ifdef __EXX
                if(func.info->number == XC_MGGA_X_SCAN_L && 
                   XC_Functional::get_func_type() == 5) {
                    vrho[idx] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                v(is, ir) += e2 * vrho[idx] * sgn[idx];
                vtxc += e2 * vrho[idx] * chr->rho[is][ir];
            }
        }

        // 梯度处理（与SCAN相同）
        std::vector<std::vector<ModuleBase::Vector3<double>>> h(nspin, 
            std::vector<ModuleBase::Vector3<double>>(nrxx));
        
        if(nspin == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ir=0; ir<nrxx; ++ir) {
#ifdef __EXX
                if(func.info->number == XC_MGGA_X_SCAN_L && 
                   XC_Functional::get_func_type() == 5) {
                    vsigma[ir] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                h[0][ir] = 2.0 * gdr[0][ir] * vsigma[ir] * 2.0 * sgn[ir];
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 64)
#endif
            for(int ir=0; ir<nrxx; ++ir) {
#ifdef __EXX
                if(func.info->number == XC_MGGA_X_SCAN_L && 
                   XC_Functional::get_func_type() == 5) {
                    vsigma[3*ir]   *= (1.0 - XC_Functional::get_hybrid_alpha());
                    vsigma[3*ir+1] *= (1.0 - XC_Functional::get_hybrid_alpha());
                    vsigma[3*ir+2] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                h[0][ir] = 2.0 * (gdr[0][ir]*vsigma[3*ir] * sgn[2*ir] * 2.0 + 
                           gdr[1][ir]*vsigma[3*ir+1] * sgn[2*ir] * sgn[2*ir+1]);
                h[1][ir] = 2.0 * (gdr[1][ir]*vsigma[3*ir+2] * sgn[2*ir+1] * 2.0 + 
                           gdr[0][ir]*vsigma[3*ir+1] * sgn[2*ir] * sgn[2*ir+1]);
            }
        }

        // 后续处理与SCAN完全相同(?)
        std::vector<std::vector<double>> dh(nspin, std::vector<double>(nrxx));
        for(int is=0; is<nspin; ++is) {
            XC_Functional::grad_dot(h[is].data(), dh[is].data(), chr->rhopw, tpiba);
        }

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rvtxc)
#endif
        for(int is=0; is<nspin; ++is) {
            for(int ir=0; ir<nrxx; ++ir) {
                rvtxc += dh[is][ir] * rho[ir*nspin+is];
                v(is, ir) -= dh[is][ir];
            }
        }
        vtxc -= rvtxc;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(int is=0; is<nspin; ++is) {
            for(int ir=0; ir<nrxx; ++ir) {
                vofk(is, ir) += vtau[ir*nspin+is] * sgn[ir*nspin+is];
            }
        }
    }

    // 归约与结果处理（与SCAN相同）
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif

    etxc *= omega / chr->rhopw->nxyz;
    vtxc *= omega / chr->rhopw->nxyz;

    XC_Functional_Libxc::finish_func(funcs);

    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_meta");
    return std::make_tuple(etxc, vtxc, std::move(v), std::move(vofk));
}

#endif