#!/usr/bin/env python3
"""
Slack 봇 아이콘 리사이징 도구
이미지를 512x512px ~ 2000x2000px 사이의 정사각형으로 자릅니다.
"""

import os
import sys
import re
from PIL import Image

def convert_windows_path(path: str) -> str:
    """
    Windows 경로를 WSL 경로로 변환합니다.
    C:\\Users\\... -> /mnt/c/Users/...
    """
    # Windows 경로 패턴 확인 (C:\ 또는 C:/)
    windows_pattern = re.match(r'^([A-Za-z]):[/\\](.*)$', path)
    if windows_pattern:
        drive_letter = windows_pattern.group(1).lower()
        rest_path = windows_pattern.group(2).replace('\\', '/')
        wsl_path = f"/mnt/{drive_letter}/{rest_path}"
        print(f"🔄 Windows 경로를 WSL 경로로 변환: {path} -> {wsl_path}")
        return wsl_path
    return path


def resize_icon(input_path: str, output_path: str = None, size: int = 512):
    """
    이미지를 정사각형으로 자르고 리사이징합니다.
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로 (None이면 자동 생성)
        size: 출력 크기 (512 ~ 2000 사이, 기본값: 512)
    """
    # 크기 검증
    if not (512 <= size <= 2000):
        print(f"❌ 오류: 크기는 512px ~ 2000px 사이여야 합니다. (입력: {size}px)")
        return False
    
    # Windows 경로를 WSL 경로로 변환
    input_path = convert_windows_path(input_path)
    if output_path:
        output_path = convert_windows_path(output_path)
    
    # 입력 파일 확인
    if not os.path.exists(input_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_path}")
        print(f"💡 팁: WSL에서는 Windows 경로를 /mnt/c/ 형식으로 변환해야 합니다.")
        return False
    
    try:
        # 이미지 열기
        img = Image.open(input_path)
        print(f"✅ 이미지 로드 완료: {img.size[0]}x{img.size[1]}px")
        
        # 정사각형으로 자르기 (중앙 기준)
        width, height = img.size
        min_dim = min(width, height)
        
        # 중앙 좌표 계산
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        # 이미지 자르기
        img_cropped = img.crop((left, top, right, bottom))
        print(f"✅ 정사각형으로 자름: {min_dim}x{min_dim}px")
        
        # 리사이징
        img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
        print(f"✅ 리사이징 완료: {size}x{size}px")
        
        # 출력 경로 설정
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.dirname(input_path) or "."
            output_path = os.path.join(output_dir, f"{base_name}_{size}x{size}.png")
        
        # PNG로 저장 (Slack 권장 형식)
        img_resized.save(output_path, "PNG", optimize=True)
        print(f"✅ 저장 완료: {output_path}")
        print(f"📊 파일 크기: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법:")
        print(f"  python {sys.argv[0]} <이미지_경로> [출력_경로] [크기]")
        print("\n예시:")
        print(f"  python {sys.argv[0]} icon.jpg")
        print(f"  python {sys.argv[0]} icon.jpg icon_512.png")
        print(f"  python {sys.argv[0]} icon.jpg icon_1024.png 1024")
        print("\n옵션:")
        print("  이미지_경로: 리사이징할 이미지 파일 경로 (필수)")
        print("  출력_경로: 저장할 파일 경로 (선택, 기본: 원본이름_크기.png)")
        print("  크기: 출력 크기 512~2000px (선택, 기본: 512)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    
    print("=" * 60)
    print("🎨 Slack 봇 아이콘 리사이징 도구")
    print("=" * 60)
    print(f"입력 파일: {input_path}")
    print(f"출력 크기: {size}x{size}px")
    print("=" * 60)
    
    success = resize_icon(input_path, output_path, size)
    
    if success:
        print("=" * 60)
        print("✅ 완료!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ 실패!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

