#include "gcopter/visualizer.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"


#include <ros/console.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include "planner/learning_planner.hpp"


struct Config
{

    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;


    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
    }
};



class PlannerServer
{
private:

    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub, targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    Trajectory<5> jerk_traj;
    Trajectory<7> snap_traj;
    double trajStamp;

    std::string mapTopic, targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;


    LearningPlanner learning_planner_;
    int optOrder;
    std::string modelPath;

public:
    PlannerServer(ros::NodeHandle &nh_, ros::NodeHandle &nh_private)
        : nh(nh_),
          mapInitialized(false),
          visualizer(nh_private),
          learning_planner_(nh_private),
          config(Config(nh_private))
    {

        nh_private.getParam("ModelPath", modelPath);
        learning_planner_.loadModel(modelPath);

        nh_private.getParam("MapTopic", mapTopic);
        nh_private.getParam("TargetTopic", targetTopic);
        nh_private.getParam("OptOrder", optOrder);

        Eigen::Vector3d map_size;
        mapBound.resize(6);

        nh_private.param("map/x_size", map_size(0), 40.0);
        nh_private.param("map/y_size", map_size(1), 40.0);
        nh_private.param("map/z_size", map_size(2), 5.0);
        nh_private.param("map/x_origin", mapBound[0], -20.0);
        nh_private.param("map/y_origin", mapBound[2], -20.0);
        nh_private.param("map/z_origin", mapBound[4], 0.0);
        nh_private.param("map/resolution", voxelWidth, 0.1);
        nh_private.param("map/inflate_radius", dilateRadius, 0.1);

        mapBound[1] = mapBound[0] + map_size(0);
        mapBound[3] = mapBound[2] + map_size(1);
        mapBound[5] = mapBound[4] + map_size(2);

        const Eigen::Vector3i xyz((mapBound[1] - mapBound[0]) / voxelWidth,
                                  (mapBound[3] - mapBound[2]) / voxelWidth,
                                  (mapBound[5] - mapBound[4]) / voxelWidth);

        const Eigen::Vector3d offset(mapBound[0], mapBound[2], mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, voxelWidth);

        mapSub = nh.subscribe(mapTopic, 1, &PlannerServer::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(targetTopic, 1, &PlannerServer::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());
    }




    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                        fdata[cur + 1],
                                                        fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            Eigen::MatrixXd iniState(3, 3), finState(3, 3);
            std::vector<Eigen::Vector3d> route;

            iniState << startGoal.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
            finState << startGoal.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

            std::vector<Eigen::MatrixXd> startEndStates;

            startEndStates.push_back(iniState);
            startEndStates.push_back(finState);           

            auto t1 = std::chrono::high_resolution_clock::now();
            printf("\033[32m ============================ New Try ===================================\n\033[0m");

            if(learning_planner_.plan(iniState, finState, route, voxelMap))
            {
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
                std::cout << "The total time is : " << fp_ms.count()  << " ms" << std::endl;


                if (optOrder == 3)
                {
                    learning_planner_.getTraj(jerk_traj);
                    visualizer.visualize(jerk_traj, route);
                }else
                {
                    learning_planner_.getTraj(snap_traj);
                    visualizer.visualize(snap_traj, route);

                }

                trajStamp = ros::Time::now().toSec();

            }

            std::vector<Eigen::MatrixX4d> plys;
            learning_planner_.gethPolys(plys);
            visualizer.visualizePolytope(plys);

        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = mapBound[4] + dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (mapBound[5] - mapBound[4] - 2 * dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.25, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));


        if (optOrder == 3)
        {

            if (jerk_traj.getPieceNum() > 0)
            {
                const double delta = ros::Time::now().toSec() - trajStamp;
                if (delta > 0.0 && delta < jerk_traj.getTotalDuration())
                {
                    double thr;
                    Eigen::Vector4d quat;
                    Eigen::Vector3d omg;

                    flatmap.forward(jerk_traj.getVel(delta),
                                    jerk_traj.getAcc(delta),
                                    jerk_traj.getJer(delta),
                                    0.0, 0.0,
                                    thr, quat, omg);
                    double speed = jerk_traj.getVel(delta).norm();
                    double bodyratemag = omg.norm();
                    double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                    std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                    speedMsg.data = speed;
                    thrMsg.data = thr;
                    tiltMsg.data = tiltangle;
                    bdrMsg.data = bodyratemag;
                    visualizer.speedPub.publish(speedMsg);
                    visualizer.thrPub.publish(thrMsg);
                    visualizer.tiltPub.publish(tiltMsg);
                    visualizer.bdrPub.publish(bdrMsg);

                    visualizer.visualizeSphere(jerk_traj.getPos(delta),
                                            dilateRadius);
                }
            }
        }else 
        {

            if (snap_traj.getPieceNum() > 0)
            {
                const double delta = ros::Time::now().toSec() - trajStamp;
                if (delta > 0.0 && delta < snap_traj.getTotalDuration())
                {
                    double thr;
                    Eigen::Vector4d quat;
                    Eigen::Vector3d omg;

                    flatmap.forward(snap_traj.getVel(delta),
                                    snap_traj.getAcc(delta),
                                    snap_traj.getJer(delta),
                                    0.0, 0.0,
                                    thr, quat, omg);
                    double speed = snap_traj.getVel(delta).norm();
                    double bodyratemag = omg.norm();
                    double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                    std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                    speedMsg.data = speed;
                    thrMsg.data = thr;
                    tiltMsg.data = tiltangle;
                    bdrMsg.data = bodyratemag;
                    visualizer.speedPub.publish(speedMsg);
                    visualizer.thrPub.publish(thrMsg);
                    visualizer.tiltPub.publish(tiltMsg);
                    visualizer.bdrPub.publish(bdrMsg);

                    visualizer.visualizeSphere(snap_traj.getPos(delta),
                                            dilateRadius);
                }

            }


        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "planning_node");
    ros::NodeHandle nh_, nh_private("~");

    PlannerServer planner_server_utils(nh_, nh_private);

    ros::Rate lr(1000);
    while (ros::ok())
    {
        planner_server_utils.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}

