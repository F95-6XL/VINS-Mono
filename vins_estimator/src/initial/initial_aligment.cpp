#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    // 我们有两个观测源，一个是相机，得到帧的绝对坐标
    // 另一个是IMU，得到帧间的相对坐标
    // 因为IMU中的bg没有得到过校准，IMU误差较大，所以使用视觉结果校准IMU
    // 注意由于纯视觉得到的坐标没有scale，这里暂时只能用sfm得到的旋转

    // 假设有相邻两帧i和j，通过之前PnP得到了两帧的旋转qi，qj
    // 那么qij = qi^-1 * qj
    // 此外, 我们还有IMU积分得到的ij间的位姿，保存在j帧的预积分结果中，令为gamma_ij
    // 那么理论上有gamma_ij = qi^-1 * qj (1)
    // 注意，严格意义上上面的旋转都还要加上sensor_to_body的旋转，
    // 但是这里因为是相对旋转，就不需要了

    // 现在考虑实际情况，引入由于bw的微小误差引起的IMU积分量gamma_ij的误差变化
    // 这里我们采用一阶近似，即
    // d_gamma_ij = J_gamma_bw * dbw
    // 根据四元数的性质，我们又知道对于微小的旋转角轴theta，有q(theta) = [1 1/2*theta]
    // 因此，gamma_ij真值与测量值存在如下转换关系：
    // gamma_ij = ^gamma_ij * [1 1/2 * J_gamma_bw * dbw] (2)，
    // 其中^gamma_ij代表测量值，[1 1/2 * J_gamma_bw * dbw]为bw所引起的四元数的误差
    // 将(2)带入(1)式，有
    // ^gamma_ij * [1 1/2 * J_gamma_bw * dbw] = qi^-1 * qj
    // 在vins中，我们接下来只关注四元数的虚部，即
    // ^gamma_ij * 1/2 * J_gamma_bw * dbw = qi^-1 * qj
    // 移项得
    // ^gamma_ij^-1 * ^gamma_ij * dbw = 2 * J_gamma_bw^-1 * ^gamma_ij^-1 * qi^-1 * qj (3)

    // 式(3)有Ax=b的形式。其中 x=dbw 为待求的偏差
    // 至此我们通过相邻的两帧解出了bw，但是稳定性较差
    // 因此我们对所有相邻帧求bw，并对bw求均值 

    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        // 遍历所有帧，每次都处理相邻两帧
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R); // i to j

        // pre_integration保存了相邻两帧之间的IMU中值积分结果
        // 从中取出角度对于bw的雅可比项，保存在tmp_A中
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();

        // 这里对A和B做累加，本质上就是对各帧算出来的bw求均值
        // A1 * x = B1, A2 * x = B2, ...
        // 所有式子相加，得（A1 + A2 + ...）* x = (B1 + B2 + ...)
        A += tmp_A.transpose() * tmp_A; 
        b += tmp_A.transpose() * tmp_b;

    }
    // Ax=b, x=delta_bg
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 上面求了均值，所以认为解出来的值对所有帧通用
    // 而Bgs在初始化的clearState函数中全部被置为0
    // 所以这里挨个加上就行
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        // 在得到了修正后的bw后，对所有IMU进行重新积分
        // Bgs里的每个值都相等，随便取一个就好
        // Bas则继续认为等于0
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]); 
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // g实际上只有两个自由度，因为g的模是已知的，因此这里对g进行参数化，对它进行降维
    // 假设真实的g在目前初始估计的g的邻域中
    // 先使用g的模的真值对g的初始估计进行修正，再定义两个参数乘以两个修正方向，以对初始估计的g值进行修正
    // 两个修正方向与当前g的初始估计方向形成正交系，两个修正参数就是待优化参数
    // 即
    // g_fine = |g_true| * g_unit_coarse + w1*b1 + w2*b2, 其中 b1 x b2 = g_unit_coarse, w1w2为待优化参数
    // 但是修正后的g模值就不对了，所以求完之后再进行一次模值修正


    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 再次遍历所有的相邻帧，使用IMU和视觉结果确定scale，并计算重力系数g
    // g的模认为是已知的
    // 这里关注g的原因是我们的世界坐标系是以初始化帧中的相机坐标为原点
    // 而它的z轴和真实世界坐标的z轴不一定重合，所以要对g的方向进行修正

    // 依然关注相邻两帧ij，设j帧的预积分结果为∆vij，∆pij
    // 则有
    // vj = vi − g ∗ dt + qi * ∆vij
    // pj = pi + vi ∗ dt − 1/2 * g ∗ dt^2 + qi * ∆pij (1)

    // 其中pi, pj都是世界坐标系下的IMU位姿
    // 而我们目前的位姿都是sfm得到的，是视觉的，因此有如下问题：
    // 1. 我们已知的仅有相机的位姿，而公式中的位姿是IMU的位姿 -> 引入IMU和相机之间的坐标转换
    // 2. 位姿没有scale的 -> 引入scale系数s
    // 即有 
    // s * pbk = s * pck - Rcb * pbc (2)
    // 其中pbk是世界坐标系下的IMU坐标，是目标量。pck是相机位姿，是已知量。pbc是相机IMU的标定参数。
    // 此外，(1)中的vi, vj也是未知量。因为sfm只获得位姿，IMU预积分也只获得了∆vij。
    // 总结一下，我们的未知量是各帧速度v，重力加速度g，以及scale系数s

    // 把(2)代入(1)，整理得
    // ∆vij = qi^-1 * (qj * vj_j + g * dt - qi * vi_i)
    // ∆pij = qi^-1 * (s * pbj + 1/2 * g * dt^2 - s * pbi - qi * vi_i * dt)
    //      = qi^-1 * (s * pcj - Rcb * pbc + 1/2*g*dt^2 - s * pci + Rcb * pbc - qi * vi_i * dt)
    //      = qi^-1 * s * (pcj - pci) - qi^-1 * qj * pbc + pcb - vi_i * dt + 1/2 * qi^-1 * g*dt^2 
    // 这里我们把每帧v的未知数设为了在各自IMU体坐标系下的值

    // 整理成矩阵形式
    // | -I*dt   0     1/2*qi^-1*dt^2  qi^-1(pj-pi) |  *  | vi_i vj_j g s |^T  =  | ∆pij - pbc + qi^-1 * qj * pbc |
    // |  -I  qi^-1*qj     qi^-1*dt           0     |                             |             ∆vij              |
    // 至此，相邻两帧的优化方程建立了。遍历所有相邻两帧，就可以构建方程组

    // 上面的方程有Ax = b的形式，其中A为6x10矩阵,x为10维向量，b为6维向量
    // 在下面的实现中，我们实际解的是A^T*Ax = A^T*b
    // 这样变成方阵去解，可以利用chelosky分解求解


    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1; // 状态量为N个速度 + g + s

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 上面的计算把g当作一个三维变量求解，给出了g的初始估计
    // 但我们只是不知道g的方向，g的模是知道的，即|g| = 9.81
    // 所以其实g只有二维自由度
    // 下面在g的初始位姿估计结果上引入模长这一维约束，把上面的计算再重新来一遍
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    // 目前我们各自对相机和IMU获得的信息进行了处理
    // 接下来我们比较这两者获得的信息，来求解三个未知量：陀螺仪偏差Bgs，重力加速度g和视觉尺度s
    // 这里分成两步进行：

    // 1. 通过比较视觉和IMU的结果，修正陀螺仪参数bg
    //    这里用到了视觉和IMU中的旋转q构建约束
    solveGyroscopeBias(all_image_frame, Bgs);

    // 2. 修正重力加速度g，并计算尺度s
    //    这里用到了视觉和IMU中的速度v和平移p构建约束
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
