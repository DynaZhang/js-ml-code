export const file2Blob = (file: File): Promise<Blob | BlobPart> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = (e: ProgressEvent<FileReader>) => {
            const result = e.target?.result as ArrayBuffer | BlobPart;
            let blob;
            if (typeof result === 'object') {
                blob = new Blob([result])
            } else {
                blob = result;
            }
            resolve(blob);
        };
        reader.onerror = (e: ProgressEvent<FileReader>) => {
            reject(e);
        };
    });
}

export const file2Image = (file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (e: ProgressEvent<FileReader>) => {
            const image = new Image();
            image.src = e.target?.result as string;
            image.width = 224;
            image.height = 224;
            resolve(image);
        };
        reader.onerror = (e: ProgressEvent<FileReader>) => {
            reject(e);
        }
    });
};