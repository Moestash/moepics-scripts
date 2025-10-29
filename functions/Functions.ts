import axios from "axios"
import phash from "sharp-phash"

export default class Functions {
    public static binaryToHex = (bin: string) => {
        return bin.match(/.{4}/g)?.reduce(function(acc, i) {
            return acc + parseInt(i, 2).toString(16).toUpperCase()
        }, "") || ""
    }
    
    public static imageBuffer = async (link: string, headers?: {[key: string]: string}) => {
        const response = await axios.get(link, {responseType: "arraybuffer", 
        headers: {Referer: "https://www.pixiv.net/", ...headers}}).then((r) => r.data)
        return Buffer.from(response)
    }
    
    public static pHash = async (buffer: Buffer) => {
        return phash(buffer).then((hash: string) => this.binaryToHex(hash))
    }
}