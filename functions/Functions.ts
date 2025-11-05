import Moepictures from "moepics-api"
import axios from "axios"
import phash from "sharp-phash"

const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

export default class Functions {
    public static removeDuplicates = <T>(array: T[]) => {
        const set = new Set<string>()
        return array.filter(item => {
            const serialized = JSON.stringify(item)
            if (set.has(serialized)) {
                return false
            } else {
                set.add(serialized)
                return true
            }
        })
    }
    
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

    public static getDanbooruArtistTag = async (tag: string) => {
        const posts = await moepics.search.posts({query: tag})
        for (const post of posts) {
            if (post.mirrors?.danbooru) {
                let id = post.mirrors.danbooru.match(/\d+/)?.[0]
                let danbooruPost = await fetch(`https://danbooru.donmai.us/posts/${id}.json`).then((r) => r.json())
                if (danbooruPost.tag_string_artist?.split(" ").length > 1) continue
                const danbooruArtistTag = danbooruPost.tag_string_artist?.split(" ")[0]
                return danbooruArtistTag as string
            }
        }
        return ""
    }
}