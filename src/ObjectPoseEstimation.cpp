/**
 * @Brief:   3D Object Pose Estimation
 * @Author:  Shengchao Niu
 * @Email:   niushengchao@gmail.com
 * @Blog:    http://hjc-niu.blogspot.com
 * @Github:  https://github.com/hjc-niu
 * @Date:    2023-12-02
 */

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/console/parse.h>
// correspondence represents a match between two entities
#include <pcl/correspondence.h>
// estimates local surface properties at each 3D point,
// such as surface normals and curvatures, in parallel,
// using the OpenMP standard.
#include <pcl/features/normal_3d_omp.h>
// estimates the Signature of Histograms of Orientations (SHOT) descriptor
// for a given point cloud data set containing points and normals,
// in parallel, using the OpenMP standard.
#include <pcl/features/shot_omp.h>
// assembles a local 3D grid over a given PointCloud,
// and down samples + filters the data
#include <pcl/filters/uniform_sampling.h>
// a 3D voting space
#include <pcl/recognition/cg/hough_3d.h>
// a 3D correspondence grouping enforcing geometric consistency
// among feature correspondences
#include <pcl/recognition/cg/geometric_consistency.h>
// a hypothesis verification method proposed
// in "A Global Hypotheses Verification Method for 3D Object Recognition"
#include <pcl/recognition/hv/hv_go.h>
// PCL Visualizer main class
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/cloud_viewer.h>
// a generic type of 3D spatial locator using kD-tree structures
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// allows us to use pcl::transformPointCloud function
#include <pcl/common/transforms.h>

struct PointCloudStyle
{
    // color
    double dlRed;
    double dlGreen;
    double dlBlue;

    // size
    double dlSize;

    PointCloudStyle(double dlRed, double dlGreen,   double dlBlue,  double dlSize):
                    dlRed(dlRed), dlGreen(dlGreen), dlBlue(dlBlue), dlSize(dlSize){}
};

PointCloudStyle g_pcsWhite( 255.0, 255.0, 255.0, 4.0);
PointCloudStyle g_pcsRed(   255.0,   0.0,   0.0, 3.0);
PointCloudStyle g_pcsGreen(   0.0, 255.0,   0.0, 5.0);
PointCloudStyle g_pcsBlue(    0.0,   0.0, 255.0, 4.0);
PointCloudStyle g_pcsPurple(255.0,   0.0, 255.0, 8.0);

// Whether to show key points
bool g_bIsShowKeyPoints       = false;
// Whether to enable clutter detect
bool g_bIsEnableClutterDetect = false;

// Model uniform sampling radius
float g_fModelSamplingRadius  = 0.02f;
// Scene uniform sampling radius
float g_fSceneSamplingRadius  = 0.02f;
float g_fReferenceFrameRadius = 0.015f;
float g_fDescriptorRadius     = 0.02f;
float g_fClutterRadius        = 0.03f;
float g_fNormalsRadius        = 0.05f;

float g_fClusterThr   = 5.0f;
float g_fInlierThr    = 0.001f;
float g_fOcclusionThr = 0.001f;

// Sets the size of each bin of Hough space
float g_fClusterSize        = 0.01f;
// Iterative Closest Point
float g_fICPCorDistance     = 0.05f;
float g_fClutterRegularizer = 5.0f;
float g_fRegularizerValue   = 3.0f;

// ICP max iterations number
int g_iICPMaxIterNum = 50;

extern char* __progname;

void
show_usage()
{
    std::cout
    << "\nUsage:\n"
    << "  " << __progname << " modal_file.pcd scene_file.pcd [options]\n"
    << "Options:\n"
    << "  -h:                     Show this help\n\n"

    << "  -k                      Show key points\n"
    << "  -d                      enabled clutter detect\n\n"

    << "  --model_radius   <val>: Model uniform sampling radius (default " << g_fModelSamplingRadius  << ")\n"
    << "  --scene_radius   <val>: Scene uniform sampling radius (default " << g_fSceneSamplingRadius  << ")\n"
    << "  --ref_radius     <val>: Reference frame radius        (default " << g_fReferenceFrameRadius << ")\n"
    << "  --desc_radius    <val>: Descriptor radius             (default " << g_fDescriptorRadius     << ")\n"
    << "  --cluster_radius <val>: Clutter radius                (default " << g_fClutterRadius        << ")\n"
    << "  --normals_radius <val>: Normals radius                (default " << g_fNormalsRadius        << ")\n\n"

    << "  --cluster_thr    <val>: Clustering threshold          (default " << g_fClusterThr           << ")\n"
    << "  --inlier_thr     <val>: Inlier threshold              (default " << g_fInlierThr            << ")\n"
    << "  --occlusion_thr  <val>: Occlusion threshold           (default " << g_fOcclusionThr         << ")\n\n"

    << "  --cluster_size   <val>: Cluster size                  (default " << g_fClusterSize          << ")\n"
    << "  --icp_corr_dist  <val>: ICP correspondence distance   (default " << g_fICPCorDistance       << ")\n"
    << "  --clutter_reg    <val>: Clutter Regularizer           (default " << g_fClutterRegularizer   << ")\n"
    << "  --regularizer    <val>: Regularizer value             (default " << g_fRegularizerValue     << ")\n\n"

    << "  --max_iter_num   <val>: ICP max iterations number     (default " << g_iICPMaxIterNum        << ")\n"
    << std::endl;
}

int
parse_argv(int    argc,
           char*  argv[],
           char** ppcModelPCDPath,
           char** ppcScenePCDPath)
{
    int iResult = 0;

    do
    {
        if ((1 == argc) ||
            (pcl::console::find_switch(argc,
                                       argv,
                                       "-h")))
        {
            //std::cout << "### Info: argc = " << argc << " argv = " << *argv << " ###" << std::endl;
            show_usage();

            iResult = -1;
            break;
        }

        // at least three parameters
        if (2 >= argc)
        {
            show_usage();

            iResult = -2;
            break;
        }

        // get two paths to the Model & Scene PCD file
        std::vector<int> vecPCDPath;
        vecPCDPath = pcl::console::parse_file_extension_argument(argc,
                                                                 argv,
                                                                 ".pcd");

        if (2 != vecPCDPath.size())
        {
            std::cout << "!!! Err: two PCD files are required, the model and the scene file !!!" << std::endl;

            show_usage();

            iResult = -3;
            break;
        }

        *ppcModelPCDPath = argv[vecPCDPath[0]];
        *ppcScenePCDPath = argv[vecPCDPath[1]];

        if (pcl::console::find_switch(argc,
                                      argv,
                                      "-k"))
        {
            g_bIsShowKeyPoints = true;
        }

        if (pcl::console::find_switch(argc,
                                      argv,
                                      "-d"))
        {
            g_bIsEnableClutterDetect = true;
        }

        pcl::console::parse_argument(argc, argv, "--model_radius",   g_fModelSamplingRadius);
        pcl::console::parse_argument(argc, argv, "--scene_radius",   g_fSceneSamplingRadius);
        pcl::console::parse_argument(argc, argv, "--ref_radius",     g_fReferenceFrameRadius);
        pcl::console::parse_argument(argc, argv, "--desc_radius",    g_fDescriptorRadius);
        pcl::console::parse_argument(argc, argv, "--cluster_radius", g_fClutterRadius);
        pcl::console::parse_argument(argc, argv, "--normals_radius", g_fNormalsRadius);

        pcl::console::parse_argument(argc, argv, "--cluster_thr",    g_fClusterThr);
        pcl::console::parse_argument(argc, argv, "--inlier_thr",     g_fInlierThr);
        pcl::console::parse_argument(argc, argv, "--occlusion_thr",  g_fOcclusionThr);

        pcl::console::parse_argument(argc, argv, "--cluster_size",   g_fClusterSize);
        pcl::console::parse_argument(argc, argv, "--icp_corr_dist",  g_fICPCorDistance);
        pcl::console::parse_argument(argc, argv, "--clutter_reg",    g_fClutterRegularizer);
        pcl::console::parse_argument(argc, argv, "--regularizer",    g_fRegularizerValue);

        pcl::console::parse_argument(argc, argv, "--max_iter_num",   g_iICPMaxIterNum);

    }while(0);

    return iResult;
}

int
main(int   argc,
     char* argv[])
{
    int iResult = 0;

    do
    {
        char* pcModelPCDPath = NULL;
        char* pcScenePCDPath = NULL;

        iResult = parse_argv(argc,
                             argv,
                             &pcModelPCDPath,
                             &pcScenePCDPath);
        if (0 != iResult)
        {
            //std::cout << "!!! Err: parse_argv " << iResult << " failed !!!" << std::endl;
            iResult = -1;
            break;
        }

        // point cloud of model
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelPC(new pcl::PointCloud<pcl::PointXYZRGBA>());

        if (0 > pcl::io::loadPCDFile(pcModelPCDPath,
                                     *ptrModelPC))
        {
            std::cout << "!!! Err: loading PCD file of the model failed !!!" << std::endl;
            iResult = -2;
            break;
        }

        // point cloud of scene
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrScenePC(new pcl::PointCloud<pcl::PointXYZRGBA>());

        if (0 > pcl::io::loadPCDFile(pcScenePCDPath,
                                     *ptrScenePC))
        {
            std::cout << "!!! Err: loading PCD file of the scene failed !!!" << std::endl;
            iResult = -3;
            break;
        }

        /** 1. Compute the normals and the curvatures */
        // estimates local surface properties at each 3D point,
        // such as surface normals and curvatures, in parallel, using the OpenMP standard
        pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> objNormalEstiOMP;
        // set the number of 10 nearest neighbours to use for the feature estimation
        objNormalEstiOMP.setKSearch(10);

        objNormalEstiOMP.setInputCloud(ptrModelPC);
        // normal vector of point cloud of model
        pcl::PointCloud<pcl::Normal>::Ptr ptrModelNV(new pcl::PointCloud<pcl::Normal>());
        // output the resultant point cloud model dataset containing the estimated features
        objNormalEstiOMP.compute(*ptrModelNV);

        objNormalEstiOMP.setInputCloud(ptrScenePC);
        // normal vector of point cloud of scene
        pcl::PointCloud<pcl::Normal>::Ptr ptrSceneNV(new pcl::PointCloud<pcl::Normal>());
        objNormalEstiOMP.compute(*ptrSceneNV);

        /** 2. Obtain key points through downsampling filtering using uniform sampling */
        pcl::UniformSampling<pcl::PointXYZRGBA> objUniformSampling;
        objUniformSampling.setInputCloud(ptrModelPC);
        objUniformSampling.setRadiusSearch(g_fModelSamplingRadius);
        // point cloud of keypoints of model
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelKP(new pcl::PointCloud<pcl::PointXYZRGBA>());
        // output the resultant filtered point cloud dataset
        objUniformSampling.filter(*ptrModelKP);
        std::cout << "### Info: model >> all points: " << ptrModelPC->size()
                  << " key points: "                   << ptrModelKP->size()
                  << " ###" << std::endl;

        objUniformSampling.setInputCloud(ptrScenePC);
        objUniformSampling.setRadiusSearch(g_fSceneSamplingRadius);
        // point cloud of keypoints of scene
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrSceneKP(new pcl::PointCloud<pcl::PointXYZRGBA>());
        objUniformSampling.filter(*ptrSceneKP);
        std::cout << "### Info: scene >> all points: " << ptrScenePC->size()
                  << " key points: "                   << ptrSceneKP->size()
                  << " ###" << std::endl;

        /** 3. estimates the Signature of Histograms of OrienTations (SHOT) descriptor
         *     for a given point cloud dataset containing points and normals,
         *     in parallel, using the OpenMP standard */
        pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> objShotEstiOMP;
        objShotEstiOMP.setRadiusSearch(g_fDescriptorRadius);

        objShotEstiOMP.setInputCloud(ptrModelKP);
        objShotEstiOMP.setInputNormals(ptrModelNV);
        objShotEstiOMP.setSearchSurface(ptrModelPC);
        // feature descriptors of feature points of model point cloud
        pcl::PointCloud<pcl::SHOT352>::Ptr ptrModelDesc(new pcl::PointCloud<pcl::SHOT352>()); // SHOT352 = 32 * 11 = 352
        objShotEstiOMP.compute(*ptrModelDesc);

        objShotEstiOMP.setInputCloud(ptrSceneKP);
        objShotEstiOMP.setInputNormals(ptrSceneNV);
        objShotEstiOMP.setSearchSurface(ptrScenePC);
        // feature descriptors of feature points of scene point cloud
        pcl::PointCloud<pcl::SHOT352>::Ptr ptrSceneDesc(new pcl::PointCloud<pcl::SHOT352>());
        objShotEstiOMP.compute(*ptrSceneDesc);

        /** 4. Match two point clouds in KDTree storage mode to obtain matching point cloud groups */
        // KdTreeFLANN is a generic type of 3D spatial locator using kD-tree structures
        pcl::KdTreeFLANN<pcl::SHOT352> objKdTreeFlann;
        objKdTreeFlann.setInputCloud(ptrModelDesc);

        // represents a match between two entities
        // this is represented via the indices of a source point and a target point, and the distance between them.
        pcl::CorrespondencesPtr ptrModelSceneCorrespVec(new pcl::Correspondences());
        std::vector<int> vecModelQueryPointIdx;
        std::vector<int> vecSceneMatchPointIdx;

        for (size_t idx = 0;
                    idx < ptrSceneDesc->size();
                    idx++)
        {
            // Only be considered if the value is finite
            if (!pcl_isfinite(ptrSceneDesc->at(idx).descriptor[0])) {
                continue;
            }

            // the resultant indices of the neighbouring points
            std::vector<int>   vecNPIndices;
            // the resultant squared distances to the neighbouring points
            std::vector<float> vecNPSquaredDist; // must be resized to k a priori!

            // search for k-nearest neighbours for the given query point
            // returns number of neighbours found
            int iFoundNeighbourNum = objKdTreeFlann.nearestKSearch(ptrSceneDesc->at(idx), // a given valid query point
                                                                   // the number of neighbors to search for
                                                                   1,
                                                                   // two parameters below must be resized to k a priori!
                                                                   vecNPIndices,
                                                                   vecNPSquaredDist);
            if ((1     <= iFoundNeighbourNum) &&
                // The squared distance is added between 0 and 1
                ((1.0f >  vecNPSquaredDist[0]) &&
                 (0.0f <  vecNPSquaredDist[0])))
            {
                pcl::Correspondence stCorrespondence(vecNPIndices[0],
                                                     static_cast<int>(idx),
                                                     vecNPSquaredDist[0]);

                ptrModelSceneCorrespVec->push_back(stCorrespondence);

                // add index of the source query    point
                vecModelQueryPointIdx.push_back(stCorrespondence.index_query);
                // add index of the target matching point
                vecSceneMatchPointIdx.push_back(stCorrespondence.index_match);
            }
        }

        std::cout << "### Info: the number of 0 < squared distance < 1: " << ptrModelSceneCorrespVec->size() << " ###"
                  << std::endl;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelCorrespPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::copyPointCloud(*ptrModelKP,           // the input point cloud dataset
                            vecModelQueryPointIdx, // the vector of indices representing the points to be copied from cloud_ins
                            *ptrModelCorrespPC);   // the resultant output point cloud dataset

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrSceneCorrespPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::copyPointCloud(*ptrSceneKP,
                            vecSceneMatchPointIdx,
                            *ptrSceneCorrespPC);

        /** 5. Implements a 3D correspondence grouping algorithm */
        // a vector containing one transformation matrix for each instance of the model recognized in the scene
        // occur an error if directly using Eigen::Matrix4f in std::vector, so must use Eigen::aligned_allocator
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > vecTransformations;
        // a vector containing the correspondences for each instance of the model found within the input data
        // (the same output of clusterCorrespondences)
        std::vector<pcl::Correspondences> vecClusteredCorrs;

        // implements the border-aware repeatable directions algorithm for local reference frame estimation
        pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::ReferenceFrame> objBoardLRFE;
        // sets whether holes in the margin of the support, for each point,
        // are searched and accounted for in the estimation of the Reference Frame or not.
        objBoardLRFE.setFindHoles(true);
        objBoardLRFE.setRadiusSearch(g_fReferenceFrameRadius);

        // use keypoints to calculate reference frames
        objBoardLRFE.setInputCloud(ptrModelKP);
        objBoardLRFE.setInputNormals(ptrModelNV);
        objBoardLRFE.setSearchSurface(ptrModelPC);
        // base method for feature estimation
        // for all points given in setInputCloud(), setIndices()
        // using the surface in setSearchSurface() and the spatial locator in setSearchMethod()
        // output the resultant point cloud model dataset containing the estimated features
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr ptrModelReframePC(new pcl::PointCloud<pcl::ReferenceFrame>());
        objBoardLRFE.compute(*ptrModelReframePC);

        objBoardLRFE.setInputCloud(ptrSceneKP);
        objBoardLRFE.setInputNormals(ptrSceneNV);
        objBoardLRFE.setSearchSurface(ptrScenePC);
        pcl::PointCloud<pcl::ReferenceFrame>::Ptr ptrSceneReframePC(new pcl::PointCloud<pcl::ReferenceFrame>());
        objBoardLRFE.compute(*ptrSceneReframePC);

        // can deal with multiple instances of a model template found in a given scene
        // each correspondence casts a vote for a reference point in a 3D Hough Space
        // the remaining 3 DOFs are taken into account by associating each correspondence with a local Reference Frame
        // the suggested PointModelRfT is pcl::ReferenceFrame
        pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::ReferenceFrame, pcl::ReferenceFrame> objHough3DGrp;
        objHough3DGrp.setHoughBinSize(g_fClusterSize);
        objHough3DGrp.setHoughThreshold(g_fClusterThr);
        objHough3DGrp.setInputCloud(ptrModelKP);
        objHough3DGrp.setSceneCloud(ptrSceneKP);
        objHough3DGrp.setModelSceneCorrespondences(ptrModelSceneCorrespVec);

        objHough3DGrp.setUseInterpolation(true);
        objHough3DGrp.setUseDistanceWeight(false);
        objHough3DGrp.setInputRf(ptrModelReframePC);
        objHough3DGrp.setSceneRf(ptrSceneReframePC);

        objHough3DGrp.recognize(vecTransformations,
                                vecClusteredCorrs);

        if (0 >= vecTransformations.size())
        {
            std::cout << "!!! Err: no instance was found !!!" << std::endl;
            iResult = -4;
            break;
        }

        std::cout << "### Info: the number of instance: " << vecTransformations.size() << " ###" << std::endl;

        // generates clouds for each instance found
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr> vecInstancePC;
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr> vecTransformedPC;

        for (size_t idx = 0;
                    idx < vecTransformations.size();
                    idx++)
        {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrMatrix4fTransPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
            // transform a point cloud and rotate its normals using an Eigen transform
            pcl::transformPointCloud(*ptrModelPC,
                                     *ptrMatrix4fTransPC,
                                     vecTransformations[idx]);

            vecInstancePC.push_back(ptrMatrix4fTransPC);

            // provides a base implementation of the Iterative Closest Point algorithm
            pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> objICP;
            objICP.setInputSource(vecInstancePC[idx]);
            objICP.setInputTarget(ptrScenePC);
            // set the max correspondence distance
            objICP.setMaxCorrespondenceDistance(g_fICPCorDistance);

            // set the maximum number of iterations (criterion 1)
            objICP.setMaximumIterations(g_iICPMaxIterNum);
            // set the transformation epsilon (criterion 2)
            // (maximum allowable translation squared difference between two consecutive transformations)
            // for an optimization to be considered as having converged to the final solution.
            objICP.setTransformationEpsilon(1e-8);
            // set the euclidean distance difference epsilon (criterion 3)
            // set the maximum allowed Euclidean error between two consecutive steps in the ICP loop,
            // before the algorithm is considered to have converged
            // the error is estimated as the sum of the differences between correspondences in an Euclidean sense,
            // divided by the number of correspondences.
            objICP.setEuclideanFitnessEpsilon(1);

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrTransformedPC(new pcl::PointCloud<pcl::PointXYZRGBA>);
            // call the registration algorithm
            // which estimates the transformation and returns the transformed source (input) as output
            objICP.align(*ptrTransformedPC); // output the resultant input transformed point cloud dataset

            vecTransformedPC.push_back(ptrTransformedPC);

            std::cout << "### Info: instance[" << idx << "]";
            if (!objICP.hasConverged()) {std::cout << "not";}
            std::cout << " aligned! ###" << std::endl;
        }

        /** 6. Hypothesis Verification */
        // a global hypotheses verification method for 3D object recognition
        pcl::GlobalHypothesesVerification<pcl::PointXYZRGBA, pcl::PointXYZRGBA> objGHV;
        objGHV.setSceneCloud(ptrScenePC);
        objGHV.addModels(vecTransformedPC, true); // occlusion reasoning
        objGHV.setInlierThreshold(g_fInlierThr);
        objGHV.setOcclusionThreshold(g_fOcclusionThr);
        objGHV.setRegularizer(g_fRegularizerValue);
        objGHV.setRadiusClutter(g_fClutterRadius);
        objGHV.setClutterRegularizer(g_fClutterRegularizer);
        objGHV.setRadiusNormals(g_fNormalsRadius);
        objGHV.setDetectClutter(g_bIsEnableClutterDetect);
        objGHV.verify();

        // mask vector to identify positive hypotheses
        std::vector<bool> vecIPHMask;
        objGHV.getMask(vecIPHMask);

        for (size_t idx = 0;
                    idx < vecIPHMask.size();
                    idx++)
        {
            std::cout << "### Info: instance[" << idx << "]";
            if (!vecIPHMask[idx]) {std::cout << " not";}
            std::cout << " verified! ###" << std::endl;
        }

        /** 7. Visualization */
        pcl::visualization::PCLVisualizer objCombineVisualizer("3D Object Pose Estimation"); // the window name
        objCombineVisualizer.addPointCloud(ptrScenePC,
                                           "point cloud of scene"); // the point cloud object id

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelOffsetTransPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*ptrModelPC,
                                 *ptrModelOffsetTransPC,
                                 Eigen::Vector3f(-1, 0, 0),
                                 Eigen::Quaternionf(1, 0, 0, 0));

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelKPOffsetTransPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*ptrModelKP,
                                 *ptrModelKPOffsetTransPC,
                                 Eigen::Vector3f(-1, 0, 0),
                                 Eigen::Quaternionf(1, 0, 0, 0));

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptrModelCorOffsetTransPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*ptrModelCorrespPC,
                                 *ptrModelCorOffsetTransPC,
                                 Eigen::Vector3f(-1, 0, 0),
                                 Eigen::Quaternionf(1, 0, 0, 0));

        if (g_bIsShowKeyPoints)
        {
            // handler for predefined user colours.
            // the colour at each point will be drawn as the use given R, G, and B values.
            pcl::visualization
               ::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> objModelOffsetTransPCCHC(ptrModelOffsetTransPC,
                                                                                          g_pcsWhite.dlRed,
                                                                                          g_pcsWhite.dlGreen,
                                                                                          g_pcsWhite.dlBlue);
            objCombineVisualizer.addPointCloud(ptrModelOffsetTransPC,
                                               objModelOffsetTransPCCHC,
                                               "point cloud of model offset transform");
            // set the rendering properties of a PointCloud
            objCombineVisualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, // the property type
                                                                  g_pcsWhite.dlSize, // integer starting from 1
                                                                  "point cloud of model offset transform");
            pcl::visualization
               ::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> objModelCorOffsetTransPCCHC(ptrModelCorOffsetTransPC,
                                                                                             g_pcsPurple.dlRed,
                                                                                             g_pcsPurple.dlGreen,
                                                                                             g_pcsPurple.dlBlue);

            objCombineVisualizer.addPointCloud(ptrModelCorOffsetTransPC,
                                               objModelCorOffsetTransPCCHC,
                                               "point cloud of model corresp offset transform");
            objCombineVisualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                  g_pcsPurple.dlSize,
                                                                  "point cloud of model corresp offset transform");
            pcl::visualization
               ::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> objSceneCorPCCHC(ptrSceneCorrespPC,
                                                                                  g_pcsPurple.dlRed,
                                                                                  g_pcsPurple.dlGreen,
                                                                                  g_pcsPurple.dlBlue);

            objCombineVisualizer.addPointCloud(ptrSceneCorrespPC,
                                               objSceneCorPCCHC,
                                               "point cloud of scene corresp");
            objCombineVisualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                  g_pcsPurple.dlSize,
                                                                  "point cloud of scene corresp");
        }

        for (size_t idx = 0;
                    idx < vecInstancePC.size();
                    idx++)
        {
            std::stringstream objStringStream;
            objStringStream << "point cloud of instance " << idx;

            pcl::visualization
               ::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> objInstancePCCHC(vecInstancePC[idx],
                                                                                  g_pcsRed.dlRed,
                                                                                  g_pcsRed.dlGreen,
                                                                                  g_pcsRed.dlBlue);

            objCombineVisualizer.addPointCloud(vecInstancePC[idx],
                                               objInstancePCCHC,
                                               objStringStream.str());
            objCombineVisualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                  g_pcsRed.dlSize,
                                                                  objStringStream.str());

            PointCloudStyle pcsVerified = vecIPHMask[idx] ? g_pcsGreen : g_pcsBlue;
            objStringStream << " verified" << endl;

            pcl::visualization
               ::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> objTransformedPCCHC(vecTransformedPC[idx],
                                                                                     pcsVerified.dlRed,
                                                                                     pcsVerified.dlGreen,
                                                                                     pcsVerified.dlBlue);

            objCombineVisualizer.addPointCloud(vecTransformedPC[idx],
                                               objTransformedPCCHC,
                                               objStringStream.str());
            objCombineVisualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                  pcsVerified.dlSize,
                                                                  objStringStream.str());
        }

        pcl::visualization::PCLVisualizer objModelSceneVisualizer("compare models and scenes"); // the window name
        objModelSceneVisualizer.initCameraParameters();

        int iLeftViewPort  = 0;
        int iRightViewPort = 0;

        objModelSceneVisualizer.createViewPort(0.0f,
                                               0.0f,
                                               0.5f,
                                               1.0f,
                                               iLeftViewPort);
        objModelSceneVisualizer.setBackgroundColor(255, 255, 255,
                                                   iLeftViewPort);
        objModelSceneVisualizer.addPointCloud(ptrModelPC,
                                              "show point cloud of model",
                                              iLeftViewPort);
        objModelSceneVisualizer.addCoordinateSystem(0.1);

        objModelSceneVisualizer.createViewPort(0.5f,
                                               0.0f,
                                               1.0f,
                                               1.0f,
                                               iRightViewPort);
        objModelSceneVisualizer.setBackgroundColor(255, 255, 255,
                                                   iRightViewPort);
        objModelSceneVisualizer.addPointCloud(ptrScenePC,
                                              "show point cloud of scene",
                                              iRightViewPort);

        // returns true when the user tried to close the window
        while(!objModelSceneVisualizer.wasStopped())
        {
            // calls the interactor and updates the screen once
            objModelSceneVisualizer.spinOnce();
        }

        while(!objCombineVisualizer.wasStopped())
        {
            objCombineVisualizer.spinOnce();
        }

    }while(0);

    return iResult;
}
