import Moepictures from "moepics-api"
import axios from "axios"
import phash from "sharp-phash"
import dist from "sharp-phash/distance"
import sharp from "sharp"
import wanakana from "wanakana"
import pinyin from "pinyin"
import * as hangul from "hangul-romanization"

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

    public static cropToSquare = async (image: ArrayBuffer) => {
        const metadata = await sharp(Buffer.from(image)).metadata()
        const side = Math.min(metadata.width, metadata.height)
        const left = Math.floor((metadata.width - side) / 2)
        const top = Math.floor((metadata.height - side) / 2)
    
        const buffer = await sharp(Buffer.from(image))
            .extract({left, top, width: side, height: side})
            .toBuffer()
    
        return Object.values(new Uint8Array(buffer))
    }

    public static fixTwitterTag = (tag: string) => {
        return tag.toLowerCase().replaceAll("_", "-").replace(/^[-]+/, "").replace(/[-]+$/, "")
    }

    public static detectCJK = (text: string) => {
        const result = {
            chinese: false,
            japanese: false,
            korean: false,
            diacritics: false
        }
    
        const chineseRegex = /[\u4E00-\u9FFF]/
        const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]/
        const koreanRegex = /[\uAC00-\uD7AF\u1100-\u11FF]/
        const diacriticsRegex = /[\u00C0-\u017F\u1E00-\u1EFF\u0300-\u036F]/
    
        if (chineseRegex.test(text)) result.chinese = true
        if (japaneseRegex.test(text)) result.japanese = true
        if (koreanRegex.test(text)) result.korean = true
        if (diacriticsRegex.test(text.normalize("NFD"))) result.diacritics = true
    
        return result
    }

    public static romanizeTag = (tag: string, attributes: {chinese: boolean, japanese: boolean, korean: boolean}) => {
        let text = tag
        if (attributes.japanese) {
            text = wanakana.toRomaji(text)
        }
        if (attributes.chinese) {
            text = pinyin(text, {style: pinyin.STYLE_NORMAL}).flat().join("-")
        }
        if (attributes.korean) {
            text = hangul.convert(text)
        }
        return this.cleanTag(this.removeDiacritics(text))
    }

    public static removeDiacritics = (text: string) => {
        return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    }

    public static cleanTag = (tag: string) => {
        return tag.normalize("NFD").replace(/[^a-z0-9_\-():><&!#@?]/gi, "")
        .replaceAll("_", "-").replace(/-+/g, "-").replace(/^-+|-+$/g, "")
    }

    public static imagesMatch = async (first: string, second: string) => {
        try {
            const oldBuffer = await fetch(first).then((r) => r.arrayBuffer())
            const oldHash = await phash(Buffer.from(oldBuffer)).then((hash) => this.binaryToHex(hash))
            const buffer = await fetch(second).then((r) => r.arrayBuffer())
            const hash = await phash(Buffer.from(buffer)).then((hash) => this.binaryToHex(hash))
            if (dist(hash, oldHash) < 10) {
                return true
            }
            return false
        } catch {
            return false
        }
    }
}