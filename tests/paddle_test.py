
def paddle_ocr_test():
    from paddleocr import PaddleOCR, draw_ocr
    from PIL import Image

    ocr = PaddleOCR(lang='ch',use_angle_cls=False, use_gpu=False)  # 设置语言为中文

    image_path = r'./img/win_screenshot/win2_crop.jpg' # 被焚烧，闪烁血条  # 不够精确，需要单元测试来系统测试
    image = cv2.imread(image_path)

    num_iterations = 100
    times = []

    contains_target_text = False
    for _ in range(num_iterations):
        start_time = time.time()
        result = ocr.ocr(image, cls=False)
        for line in result:
            for box in line:
                text = box[1][0]
                if "击败" in text:
                    contains_target_text = True
                    break
            if contains_target_text:
                break
        end_time = time.time()
        times.append(end_time - start_time)
        print("击败检测结果：%s" % contains_target_text)

    # 计算总时间和平均时间
    total_time = sum(times)
    average_time = total_time / num_iterations

    print(f"Total time for {num_iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per iteration: {average_time:.4f} seconds")
    results = ocr.ocr(image, det=True)
    
    
    # 如果你想查看最后一次识别结果
    for line in result:
        print("Detected boxes and texts:")
        for box in line:
            print(f"box: {box[0]}, text: {box[1][0]}, confidence: {box[1][1]}")



if __name__ == '__main__':
    paddle_ocr_test()