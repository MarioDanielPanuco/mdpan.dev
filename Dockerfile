FROM ghcr.io/getzola/zola:v0.19.2 as zola

COPY . /myBlog
WORKDIR /myBlog
RUN ["zola", "build"]

FROM ghcr.io/static-web-server/static-web-server:2
WORKDIR /
COPY --from=zola /myBlog/public /public

# EXPOSE 8080:80


