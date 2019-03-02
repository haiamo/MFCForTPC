#pragma once
#include<time.h>//时间相关头文件，可用其中函数计算图像处理速度  
#include <stdio.h>
#include <iostream>
#include <string>

/*#include <boost/version.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#if BOOST_VERSION >= 104700
#include <boost/chrono.hpp>
#endif
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_array.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#if BOOST_VERSION >= 104900
#include <boost/interprocess/permissions.hpp>
#endif
#include <boost/iostreams/device/mapped_file.hpp>
#define BOOST_PARAMETER_MAX_ARITY 7
#include <boost/signals2.hpp>
#include <boost/signals2/slot.hpp>*/

#include <pcl\point_types.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <time.h>

#define datasize 500000

#ifndef INCLUDES_CUDAMAIN_H_
#define INCLUDES_CUDAMAIN_H_

#ifdef __cplusplus

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize,
							double* &hst_hypox, double*& hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double** &paras,
							double* &modelErr, double** &dists);

#endif

#endif