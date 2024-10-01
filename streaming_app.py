import cv2
import numpy as np

class Web_Cam():
    def __init__(self):
        print('set cam')
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow('test')
        cv2.setMouseCallback("test", self.mouseevent)

        # 클릭 좌표 저장 리스트
        self.click_positions = []

        # 상태 초기화
        self.flip_vertical = False
        self.flip_horizontal = False
        self.is_grayscale = False
        self.is_face_classification = False
        self.is_face_mosaic = False
        self.is_background_mosaic = False
        self.remove_mode = False

        classifier_path = 'C:/pr1/project_1/classifier/haarcascade_frontalface_default.xml'
        self.classifier = cv2.CascadeClassifier(classifier_path)

        # 스티커와 좌표를 저장할 리스트
        self.sticker_data = []  # (스티커, 위치, 크기) 튜플로 저장

        # 스티커 선택 변수
        self.selected_sticker = None
        self.stickers_path = {
            'smile': 'C:/pr1/project_1/emoticon_img/smile.png',
            'sad': 'C:/pr1/project_1/emoticon_img/sad.png'
        }

    def __del__(self):
        print('Release cam & destroy all windows')
        self.capture.release()
        cv2.destroyAllWindows()

    def streaming(self):
        print('Streaming...')
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 상태에 따른 프레임 변환
            if self.flip_vertical:
                frame = self.flip(frame, 0)
            if self.flip_horizontal:
                frame = self.flip(frame, 1)
            if self.is_grayscale:
                frame = self.grayscale(frame)
                # 그레이스케일에서 3채널로 변환 (BGR로 변환하여 스티커를 삽입할 수 있게 함)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if self.is_face_classification:
                frame = self.face_classification(frame)
            if self.is_face_mosaic:
                frame = self.mosaic_face(frame)
            if self.is_background_mosaic:
                frame = self.mosaic_background(frame)

            # 저장된 스티커 좌표에 스티커 삽입
            for sticker, position, size in self.sticker_data:
                frame = self.sticker(frame, sticker, position)

            cv2.imshow('test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키
                break
            elif key == ord('1'):
                self.flip_vertical = not self.flip_vertical
            elif key == ord('2'):
                self.flip_horizontal = not self.flip_horizontal
            elif key == ord('g'):
                self.is_grayscale = not self.is_grayscale
            elif key == ord('c'):
                self.is_face_classification = not self.is_face_classification
            elif key == ord('m'):
                self.is_face_mosaic = not self.is_face_mosaic
            elif key == ord('b'):
                self.is_background_mosaic = not self.is_background_mosaic
            elif key == ord('d'):
                self.remove_mode = not self.remove_mode
                if self.remove_mode:
                    print("Sticker removal mode activated. Left-click to remove stickers.")
                else:
                    print("Sticker removal mode deactivated.")

    def flip(self, frame, f):
        return cv2.flip(frame, f)

    def grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        return faces

    def face_classification(self, frame):
        faces = self.detect_face(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        return frame

    def mosaic_background(self, frame):
        faces = self.detect_face(frame)
        mosaic_frame = frame.copy()

        for (x, y, w, h) in faces:
            mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            mask[y:y+h, x:x+w] = 0

            factor = 10
            small_frame = cv2.resize(frame, (frame.shape[1] // factor, frame.shape[0] // factor))
            mosaic_background_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            mosaic_frame = cv2.bitwise_and(mosaic_background_frame, mosaic_background_frame, mask=mask)
            face_region = frame[y:y+h, x:x+w]
            mosaic_frame[y:y+h, x:x+w] = face_region

        return mosaic_frame

    def mosaic_face(self, frame):
        faces = self.detect_face(frame)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            factor = 10
            small_roi = cv2.resize(roi, (w // factor, h // factor))
            mosaic_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = mosaic_roi
        return frame

    def mouseevent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭 시 스티커 삽입
            if self.selected_sticker is not None and not self.remove_mode:
                # 현재 선택된 스티커와 좌표를 리스트에 저장
                self.sticker_data.append((self.selected_sticker, (x, y), (20, 20)))  # 추가 시 크기도 저장
            
            elif self.remove_mode:  # 스티커 제거 모드에서 클릭
                self.remove_sticker(x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:  # 우클릭 시 이미지 선택 창 열기
            self.open_image_selection_window()

    def remove_sticker(self, x, y):
        # 스티커 위치를 체크하여 제거
        for i, (sticker, position, size) in enumerate(self.sticker_data):
            pos_x, pos_y = position
            sticker_width, sticker_height = size
            
            # 스티커가 클릭된 위치에 있는지 확인
            if (pos_x <= x <= pos_x + sticker_width) and (pos_y <= y <= pos_y + sticker_height):
                del self.sticker_data[i]  # 스티커 삭제
                break

    def open_image_selection_window(self):
        # 스티커 선택 창 생성
        cv2.namedWindow('Choose Sticker')
        # 예시로 사용할 이미지 리스트 (경로는 본인 환경에 맞게 수정)
        stickers = [
            self.stickers_path['smile'],
            self.stickers_path['sad']
        ]
        sticker_images = [cv2.imread(sticker, cv2.IMREAD_UNCHANGED) for sticker in stickers]

        # 스티커 이미지를 화면에 출력
        window_height = 100
        window_width = len(sticker_images) * 100
        selection_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        for i, sticker in enumerate(sticker_images):
            resized_sticker = cv2.resize(sticker, (100, 100))  # 크기 조정
            selection_frame[:, i*100:(i+1)*100] = resized_sticker[:, :, :3]  # 스티커 삽입

        cv2.imshow('Choose Sticker', selection_frame)

        # 선택된 스티커 처리
        def sticker_selection_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭으로 스티커 선택
                index = x // 100  # 스티커 인덱스 계산
                self.selected_sticker = sticker_images[index]  # 선택한 스티커 설정
                cv2.destroyWindow('Choose Sticker')  # 선택 후 창 닫기

        # 스티커 선택 마우스 이벤트 설정
        cv2.setMouseCallback('Choose Sticker', sticker_selection_event)

    def sticker(self, frame, sticker, position):
        # 스티커 크기를 20x20으로 리사이즈
        sticker = cv2.resize(sticker, (20, 20))

        # 알파 채널 처리
        alpha_channel = sticker[:, :, 3]  # 알파 채널
        bgr_sticker = sticker[:, :, :3]  # BGR 채널

        # 스티커를 흑백으로 변환 (만약 그레이스케일 상태라면)
        if self.is_grayscale:
            bgr_sticker = cv2.cvtColor(bgr_sticker, cv2.COLOR_BGR2GRAY)
            # 흑백 이미지를 다시 3채널로 변환 (BGR로 변환, 그레이스케일 유지)
            bgr_sticker = cv2.cvtColor(bgr_sticker, cv2.COLOR_GRAY2BGR)

        # 스티커 크기
        h, w = bgr_sticker.shape[:2]

        # 클릭된 위치가 스티커의 중심이 되도록 좌표 조정
        x, y = position
        x = x - w // 2
        y = y - h // 2

        # 알파 채널을 이용해 마스크 생성
        mask = alpha_channel / 255.0
        inverse_mask = 1.0 - mask

        # 스티커와 원본 프레임을 합성
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (mask * bgr_sticker[:, :, c] + inverse_mask * frame[y:y+h, x:x+w, c])

        return frame


if __name__ == '__main__':
    cam = Web_Cam()
    cam.streaming()