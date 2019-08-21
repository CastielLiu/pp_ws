#coding: utf-8
import numpy as np
import cv2
import math




def getTransformMatrix(leftImage,rightImage):
	surf=cv2.xfeatures2d.SURF_create(400) #将Hessian Threshold设置为400,阈值越大能检测的特征就越少
	kp1,des1=surf.detectAndCompute(leftImage,None)  #查找关键点和描述符
	kp2,des2=surf.detectAndCompute(rightImage,None)

	FLANN_INDEX_KDTREE=0   #建立FLANN匹配器的参数
	indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) #配置索引，密度树的数量为5
	searchParams=dict(checks=50)    #指定递归次数
	#FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
	flann=cv2.FlannBasedMatcher(indexParams,searchParams)  #建立匹配器
	matches=flann.knnMatch(des1,des2,k=2)  #得出匹配的关键点

#	a = cv2.drawMatchesKnn(leftImage, kp1, rightImage, kp2, matches,None, flags=2)
#	cv2.namedWindow("mathches",1)
#	cv2.imshow("mathches",a)
#	cv2.waitKey()

	goodMatches=[]
	#提取优秀的特征点
	for m,n in matches:
		if m.distance < 0.45*n.distance: #如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
			goodMatches.append(m)

	#result = cv2.drawMatches(leftImage, kp1, rightImage, kp2, good,None, flags=2)
	#cv2.namedWindow("result",1)
	#cv2.imshow("result",result)
	#cv2.waitKey()

	src_pts = np.array([ kp1[m.queryIdx].pt for m in goodMatches])    #查询图像的特征描述子索引
	dst_pts = np.array([ kp2[m.trainIdx].pt for m in goodMatches])    #训练(模板)图像的特征描述子索引

	#H=cv2.findHomography(src_pts,dst_pts)         #生成变换矩阵
	Hinv = cv2.findHomography(dst_pts,src_pts)		#rightImage to leftImage
	return Hinv[0]

def mosaic(leftImage,rightImage):
	
	transformMatrix = getTransformMatrix(leftImage,rightImage) # 求解rightImage转到leftImage视角下的变换矩阵
	
	h_l, w_l = leftImage.shape[:2]
	h_r, w_r = rightImage.shape[:2]

	#右图变换到左图视角后的图片
	rImgInleftView=cv2.warpPerspective(rightImage,transformMatrix,(w_l+w_r ,h_l))#透视变换，新图像可容纳完整的两幅图
	
	#简单相加求出结果,在左右图光线明显不同的情况下，拼接后有明显痕迹
#	rImgInleftView[:,:w_l] = leftImage
#	return rImgInleftView
	
	#使用线性渐变拼接图片，像素为0的点用左图对应点填充
	#变换后的左侧顶点坐标
	leftbottom = np.dot(transformMatrix, [[0],[h_r],[1]])
	leftup = np.dot(transformMatrix, [0,0,1])
	x_leftboottom = leftbottom[0]/leftbottom[2]
	x_leftup = leftup[0]/leftup[2]
	x_start = int(max(x_leftboottom,x_leftup))
	x_end = w_l
	
	cv2.imshow('rImgInleftView',rImgInleftView)
	emptyIndex = np.nonzero(np.sum(rImgInleftView[:,x_start:x_end],axis=2)==0)
	emptyIndex = list(emptyIndex)
	emptyIndex[0] = list(emptyIndex[0])
	emptyIndex[1] = list(np.array((emptyIndex[1])) + (x_start))
	
	#emptyIndex[0] = emptyIndex[0]+x_start  #emptyIndex is tuple

	#拷贝不重叠部分
	rImgInleftView[:h_l,:x_start] = leftImage[:h_l,:x_start]
	
	dst_corners = rImgInleftView
	
	#线性渐变
	for col in range(x_start,x_end):
		k2 = float(col-x_start)/(x_end-x_start)
		k1 = 1.0-k2
		dst_corners[:,col] = k1*leftImage[:,col] + k2 * dst_corners[:,col]
	
	#重叠区黑点用左侧图填充
	dst_corners[emptyIndex] = leftImage[emptyIndex]
	
	return dst_corners

def main():
	leftImage = cv2.imread('../image/1.jpg')
	rightImage = cv2.imread('../image/2.jpg')
	result = mosaic(leftImage,rightImage)
	cv2.imshow('result',result)
	cv2.waitKey()


if __name__ == "__main__":
	main()



