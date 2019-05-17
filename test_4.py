import cv2
import time
import numpy as np

#######기본 설정 값#######
DELAY = 0.02
USE_CAM = 1
IS_FOUND = 0

MORPH = 7
CANNY = 250
##################
# 420x600 margin 10.0
_width  = 600.0
_height = 420.0
_margin = 10.0
##################

if USE_CAM: video_capture = cv2.VideoCapture(1) #webcam이 1번이기 때문에 1을 입력

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

while True :

	if USE_CAM :
		ret, rgb = video_capture.read()
	else :
		ret = 1
		rgb = cv2.imread( "opencv.jpg", 1 )

	if ( ret ):

		gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY ) # gray로 색상을 바꿈
		gray = cv2.bilateralFilter( gray, 1, 10, 120 ) # 이미지 필터링
		edges  = cv2.Canny( gray, 10, CANNY )          # 가장자리 검춤 
		kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )
		closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )
		#contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
		contours, h = cv2.findContours( closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE ) #
                #Contour
		# 2번째 인자는 cv2.RETR_LIST는 계층 구조를 고려하지 않고 contour만 추출하는 방식
		# 3번째 인자는 cv2.CHAIN_APPROX_NONE는 contour를 구성하는 모든점을 저장하는 방식

		for cont in contours:

			
			if (cv2.contourArea( cont ) > 50000) & (cv2.contourArea( cont ) < 200000) :  #너무 큰 사각형이나 작은 사각형은 무시한다.
				arc_len = cv2.arcLength( cont, True )
				approx = cv2.approxPolyDP( cont, 0.02 * arc_len, True )
				#다각형을 대상으로 꼭지점을 점점 줄여나가는 함수 
				#parameter 1. 꼭지점의 개수를 줄일 contour
				#parameter 2. 줄여나갈 오차 
				#parameter 3. 닫힌 곡선인지 열린 곡선인지

				if ( len( approx ) == 4 ): #시긱형의 경우 len(approx) == 4
					IS_FOUND = 1 #사각형을 찾은 경우 IS_FOUND가 1로 설정
					
					pts_src = np.array( approx, np.float32 )
					
					# 4개의 꼭지점을 출력
					print("result\n")
					print(pts_src)

					h, status = cv2.findHomography( pts_src, pts_dst )
					out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) ) 
					
					cv2.drawContours( rgb, [approx], -1, ( 255, 0, 0 ), 3) # 255,0,0 으로 파랑색으로 경계를 설정

				else : pass

		# 확인하기 위한 화면 2개를 띄움 
		cv2.imshow( 'gray', gray )
		cv2.imshow( 'rgb', rgb )

		if IS_FOUND :
			cv2.namedWindow( 'out', cv2.WINDOW_AUTOSIZE )
			cv2.imshow( 'out', out )
			#사각형 모양으로 잘려진 이미지 출력

		# q를 누르는 경우 
		# 프로그램 종료 
		if cv2.waitKey(27) & 0xFF == ord('q') :
			break

		# c를 누르는 경우 
		# 화면을 하여 파일로 저장 
		if cv2.waitKey(99) & 0xFF == ord('c') :
			current = str( time.time() )
			#cv2.imwrite( 'ocvi_' + current + '_edges.jpg', edges )
			#cv2.imwrite( 'ocvi_' + current + '_gray.jpg', gray )
			#cv2.imwrite( 'ocvi_' + current + '_org.jpg', rgb )
			cv2.imwrite( current + '_out.jpg', out )
			print ("Pictures saved")

		#time.sleep( DELAY )

	else :
		print ("Stopped")
		break

if USE_CAM : video_capture.release()
cv2.destroyAllWindows()

# end
