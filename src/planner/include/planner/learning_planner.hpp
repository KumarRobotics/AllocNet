#include "gcopter/trajectory.hpp"
#include "gcopter/sfc_gen.hpp"
#include <torch/script.h>
#include "qp_solver.hpp"

using namespace torch::indexing;


class LearningPlanner
{
private:
    Trajectory<5> jerk_traj;
    Trajectory<7> snap_traj;

    torch::jit::Module minsnap_conv_lstm_network;
    std::vector<torch::jit::IValue> inputs;
    torch::Device device;

    int optOrder, modelMaxSeg, seg;

    std::vector<Eigen::MatrixX4d> hPolys, vishPolys;

    QPSolver qp_solver;

    Eigen::MatrixXf eigen_stacked_hpolys;

public:
    LearningPlanner(ros::NodeHandle &nh)
        : device(torch::kCPU), // change to torch::kCPU if you want to run inference on the CPU
          qp_solver(QPConfig(nh))
    {

        nh.param("ModelMaxSeg", modelMaxSeg, 5);
        nh.param("OptOrder", optOrder, 3);

        qp_solver.setOrder(optOrder);

        std::cout << "[set up the model] optOrder  " << optOrder << std::endl;

        eigen_stacked_hpolys.resize(modelMaxSeg, 4  * 50);
    }

    inline void gethPolys(std::vector<Eigen::MatrixX4d> &plys)
    {
        plys = vishPolys;
    }

    inline void getTraj(Trajectory<5> &traj)
    {
        traj = jerk_traj;
    }

    inline void getTraj(Trajectory<7> &traj)
    {
        traj = snap_traj;
    }

    inline bool loadModel(std::string &modelPath)
    {
        // Check file existence
        std::ifstream ifile(modelPath);
        if (!ifile)
        {
            std::cerr << "Model file not found\n";
            return false;
        }
        // Deserialize the ScriptModule
        torch::autograd::AnomalyMode::set_enabled(true);
        try
        {
            minsnap_conv_lstm_network = torch::jit::load(modelPath);
            minsnap_conv_lstm_network.to(device); // Move the model to the correct device
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error loading the model\n";
            std::cerr << "Error: " << e.what() << '\n'; // Print the error message for more detail
            return false;
        }

        std::cout << "model loaded\n";
        // Generate fake input
        inputs.clear();

        torch::Tensor stacked_state = torch::randn({1, 9, 2}, torch::kFloat32).to(device);
        torch::Tensor stacked_hpolys = torch::randn({1, 50, 4, modelMaxSeg}, torch::kFloat32).to(device);

        inputs.push_back(stacked_state);
        inputs.push_back(stacked_hpolys);

        torch::Tensor output;
        // Warm up the model
        try
        {
            output = minsnap_conv_lstm_network.forward(inputs).toTensor();
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error executing the warm-up pass\n";
            std::cerr << "Error: " << e.what() << '\n'; // Print the error message for more detail
            return false;
        }

        try
        {
            output = minsnap_conv_lstm_network.forward(inputs).toTensor();
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error executing the warm-up pass\n";
            std::cerr << "Error: " << e.what() << '\n'; // Print the error message for more detail
            return false;
        }

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Perform inference
        try
        {
            output = minsnap_conv_lstm_network.forward(inputs).toTensor();
            std::cout << output << '\n';
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error executing the model\n";
            std::cerr << "Error: " << e.what() << '\n'; // Print the error message for more detail
            return false;
        }

        // Stop timing
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate duration
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        std::cout << "Time taken for model execution: " << duration_ms.count() << " ms\n";

        return true;
    }

    inline bool callModel(const Eigen::MatrixXd &iniPVA,
                          const Eigen::MatrixXd &finPVA)
    {
        inputs.clear();

        auto t1 = std::chrono::high_resolution_clock::now();

        Eigen::MatrixXf v1, v2;
        v1 = (iniPVA.transpose()).cast<float>(); // Copy #1
        v1.resize(1, 9);                         // No copy
        v2 = (finPVA.transpose()).cast<float>(); // Copy #1
        v2.resize(1, 9);                         // No copy

        Eigen::MatrixXf eigen_stacked_state(2, 9);
        eigen_stacked_state << v1, v2;
        torch::Tensor stacked_state = torch::from_blob(eigen_stacked_state.data(), {1, 9, 2}).to(device);

        eigen_stacked_hpolys.setZero();
        size_t row_num;
        Eigen::MatrixXd poly;
        for (size_t i = 0; i < seg; i++)
        {
            poly = hPolys[i].transpose();
            row_num = poly.cols();
            poly.resize(1, 4 * row_num);
            eigen_stacked_hpolys.block(i, 0, 1, 4 * row_num) = poly.cast<float>();
        } 

        torch::Tensor stacked_hpolys = torch::from_blob(eigen_stacked_hpolys.data(), {1, 50, 4, modelMaxSeg}).to(device);

        inputs.push_back(stacked_state);
        inputs.push_back(stacked_hpolys);

        // Perform inference
        torch::Tensor output_time = minsnap_conv_lstm_network.forward(inputs).toTensor();
        torch::Tensor T2 = output_time.to(at::kCPU);

        std::cout << "output_time " << output_time << std::endl;

        Eigen::Map<Eigen::VectorXf> times(T2.data_ptr<float>(), 5);

        for (size_t i = 0; i < seg; i++)
        {   
            if (times(i) < 1e-10)
            {
                std::cout << "time and seg does not fit, the segment is" << seg << std::endl;
                return false;
            }

        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = t2 - t1;

        std::cout << "2. The TPNET data load and inference time is : " << duration_ms.count() << " ms" << std::endl;
        Eigen::VectorXd flatten_coffmats;
        if (!qp_solver.solve(iniPVA, finPVA, hPolys, times, flatten_coffmats))
        {
            return false;
        }

        // p(t) = c5*t^5 + c4*t^4 + ... + c1*t + c0
        size_t idx;
        if (optOrder == 3)
        {
            jerk_traj.clear();
            jerk_traj.reserve(seg);
            Eigen::Matrix<double, 3, 6> coffMat;
            for (size_t i = 0; i < seg; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    idx = i * 3 * 6 + j * 6;
                    coffMat.row(j) = flatten_coffmats.segment(idx, 6);
                }
                jerk_traj.emplace_back(times(i), coffMat);
            }
        }
        else
        {
            snap_traj.clear();
            snap_traj.reserve(seg);
            Eigen::Matrix<double, 3, 8> coffMat;
            for (size_t i = 0; i < seg; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    idx = i * 3 * 8 + j * 8;
                    coffMat.row(j) = flatten_coffmats.segment(idx, 8);
                }

                snap_traj.emplace_back(times(i), coffMat);
            }
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        duration_ms = t3 - t2;
        std::cout << "3. The solver matrix load, solve and traj data filling time is : " << duration_ms.count() << " ms" << std::endl;

        return true;
    }


    template <typename Map>
    inline bool plan(Eigen::MatrixXd &iniState,
                     Eigen::MatrixXd &finState,
                     std::vector<Eigen::Vector3d> &route,
                     Map &mapPtr)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        if (route.size() <= 0)
        {
            sfc_gen::planPath(iniState.col(0),
                                finState.col(0),
                                mapPtr.getOrigin(),
                                mapPtr.getCorner(),
                                &mapPtr, 0.01,
                                route);
            if (route.size() <= 0)
            {
                return false;
            }
        }

        finState.col(0) = route.back();

        /* corridor generation */
        std::vector<Eigen::Vector3d> pc;
        mapPtr.getSurf(pc);

        hPolys.clear();
        vishPolys.clear();

        sfc_gen::convexCover(route,
                             pc,
                             mapPtr.getOrigin(),
                             mapPtr.getCorner(),
                             7.0,
                             3.0,
                             vishPolys);

        sfc_gen::shortCut(vishPolys);
        hPolys = vishPolys;
        seg = hPolys.size();


        if (seg > modelMaxSeg)
        {
            std::cout << "give up this try, long corridor " << std::endl;
            return false;
        }

        for (size_t i = 0; i < seg; i++)
        {
            const Eigen::ArrayXd norms =
                hPolys[i].leftCols<3>().rowwise().norm();
            hPolys[i].array().colwise() /= norms;
            hPolys[i].col(3) = -hPolys[i].col(3); // to make it work
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = t2 - t1;
        std::cout << "1. The path search and corridor generation time is : " << duration_ms.count() << " ms" << std::endl;

        return callModel(iniState, finState);
    }
};
